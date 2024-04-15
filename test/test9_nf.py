import sys
import os

import torch
import torch.nn as nn

sys.path.append(os.path.abspath("../"))

from sampler.model import FlowTransform, BaseNormalizingFlow
from test_common_helper import Feedforward, MultiGauss
from sampler._common import Distribution
from sampler.base import importance_sampling
from torch.distributions.multivariate_normal import MultivariateNormal
# test a single transform block
dim = 4
mg = MultiGauss(mean=[0] * dim, std=[1] * dim)
flowtransform = FlowTransform(dim=dim,
                              keep_dim=[0, 2],
                              scale_net=Feedforward([max(1, dim // 2), 2, 2, max(1, dim // 2)], 'relu'),
                              shift_net=Feedforward([max(1, dim // 2), 2, 2, max(1, dim // 2)], 'relu'),
                              p_base= mg)

print(flowtransform.p_base)
flowtransform.p_base = None
print(flowtransform.p_base)
flowtransform.p_base = mg
print(flowtransform.p_base)

x = torch.rand(10, dim)
x_, diff_log_det = flowtransform.backward(*flowtransform.forward(x, 0))
diff = x - x_
print(f"diff = {torch.max(torch.abs(diff))}")
print(diff)
print(f"diff_log_det = {torch.max(torch.abs(diff_log_det))}")

samples, log_prob = flowtransform.sample(10)
flowtransform.p_base = None
try:
    samples, log_prob = flowtransform.sample(10)
except Exception as e:
    print("Error raised as expected, w/ error: ", e)
samples, log_prob = flowtransform.sample(10, p_given = mg)

# test if flowtransform is compatible with IS
flowtransform.p_base = mg
results, _ = importance_sampling(10000, flowtransform, flowtransform, lambda x: x)


# after run IS, test again if flowtransform is still working (backward, forward, sample)
x = torch.rand(10, dim)
x_, diff_log_det = flowtransform.backward(*flowtransform.forward(x, 0))
diff = x - x_
print(f"diff = {torch.max(torch.abs(diff))}")
print(diff)
print(f"diff_log_det = {torch.max(torch.abs(diff_log_det))}")
samples, log_prob = flowtransform.sample(10)


try:
    results, _ = importance_sampling(10000, flowtransform, flowtransform, lambda x: x)
except Exception as e:
    print("Error raised as expected, w/ error: ", e)
    flowtransform.restore() # restore because IS is interrupted in the middle, so we need to restore the model to its original status manually. This won't be needed in the real-world use cases.

# # test a flow model
# num_trans = 4
# nf = BaseNormalizingFlow(dim=dim,
#                          num_trans=num_trans,
#                          scale_net=nn.ModuleList(
#                              [Feedforward([max(1, dim // 2), 2, 2, max(1, dim // 2)], 'relu') for _ in
#                               range(num_trans)]),
#                          shift_net=nn.ModuleList(
#                              [Feedforward([max(1, dim // 2), 2, 2, max(1, dim // 2)], 'relu') for _ in
#                               range(num_trans)]),
#                          keep_dim=[[0, 2], [1, 3], [0, 2], [1, 3]],
#                          p_base = mg)
#
# y = torch.rand(10, dim)
# y_, diff_log_det = nf.backward(*nf.forward(y, 0))
# diff = y - y_
# print(f"diff = {torch.max(torch.abs(diff))}")
# print(diff)
# print(f"diff_log_det = {torch.max(torch.abs(diff_log_det))}")
# nf.p_base = MultiGauss(mean=[0, 0, 0, 0], std=[1, 1, 1, 1])
# samples, log_prob = nf.sample(10)
#
# # test if nf is compatible with IS
#
#
#
#
#
#
# print(nf.p_base)

# results, _ = importance_sampling(10000, nf, nf, lambda x: x)