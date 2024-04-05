import sys
import os

import torch
import torch.nn as nn

sys.path.append(os.path.abspath("../"))

from sampler.model import FlowTransform, BaseNormalizingFlow
from test_common_helper import Feedforward

# test a single transform block
dim = 4
flowtransform = FlowTransform(dim=dim,
                              keep_dim=[0, 2],
                              scale_net=Feedforward([max(1, dim // 2), 2, 2, max(1, dim // 2)], 'relu'),
                              shift_net=Feedforward([max(1, dim // 2), 2, 2, max(1, dim // 2)], 'relu'))

x = torch.rand(10, dim)
x_, diff_log_det = flowtransform.backward(*flowtransform.forward(x, 0))
diff = x - x_
print(f"diff = {torch.max(torch.abs(diff))}")
print(diff)
print(f"diff_log_det = {torch.max(torch.abs(diff_log_det))}")

# test a flow model
num_trans = 4
nf = BaseNormalizingFlow(dim=dim,
                         num_trans=num_trans,
                         scale_net=nn.ModuleList(
                             [Feedforward([max(1, dim // 2), 2, 2, max(1, dim // 2)], 'relu') for _ in
                              range(num_trans)]),
                         shift_net=nn.ModuleList(
                             [Feedforward([max(1, dim // 2), 2, 2, max(1, dim // 2)], 'relu') for _ in
                              range(num_trans)]),
                         keep_dim=[[0, 2], [1, 3], [0, 2], [1, 3]]
                         )

y = torch.rand(10, dim)
y_, diff_log_det = nf.backward(*nf.forward(y, 0))
diff = y - y_
print(f"diff = {torch.max(torch.abs(diff))}")
print(diff)
print(f"diff_log_det = {torch.max(torch.abs(diff_log_det))}")
