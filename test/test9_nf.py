import sys
import os

import torch
import torch.nn as nn

sys.path.append(os.path.abspath("../"))

from sampler.model import FlowTransform, BaseNormalizingFlow
from test_common_helper import Feedforward
from sampler._common import Distribution

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

# test if nf is compatible with IS
from sampler.base import importance_sampling

class MultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(self.mean)
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
                    torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
                    + torch.log(2 * torch.pi * self.std * self.std).sum()
            )


nf.p_base = MultiGauss(mean=[0, 0, 0, 0], std=[1, 1, 1, 1])
results, _ = importance_sampling(10000, nf, nf, lambda x: x)