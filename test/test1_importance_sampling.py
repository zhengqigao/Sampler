import numpy as np
import torch
import sys

sys.path.append("D:\Coding\py\Sampler")
from sampler.base import *
from sampler._common import Distribution
from typing import Optional
from torch.distributions.multivariate_normal import MultivariateNormal
from sampler.distribution import Wrapper

test_mean = torch.Tensor([-1, 1, 0.5])


class MultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(mean)
        self.norm = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        if in_log:
            return -0.5 * (
                torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
                + torch.log(2 * torch.pi * self.std * self.std).sum()
            )
        else:
            return torch.exp(
                -0.5 * torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
            ) / (
                torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim
                * torch.prod(self.std)
            )


class MultiGaussDenser(Distribution):
    # same as MultiGauss, but evaluate_density() returns 33.3x larger value
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(mean)
        self.norm = 1 / 33.3

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        if in_log:
            return -0.5 * (
                torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
                + torch.log(2 * torch.pi * self.std * self.std).sum()
            ) + torch.log(torch.Tensor([33.3]))
        else:
            return (
                torch.exp(-0.5 * torch.sum(((x - self.mean) / self.std) ** 2, dim=1))
                / (
                    torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim
                    * torch.prod(self.std)
                )
                * 33.3
            )


# MultiGauss
# normal case
target = MultiGauss(mean=test_mean, std=[1, 1, 1])
proposal = MultiGauss(mean=[0, 0, 0], std=[1, 1, 1])
proposal.norm = 1.0
results = importance_sampling(10000, target, proposal, lambda x: x)
print("Test mean:", results)  # it works

# self.norm = None, actually 1.0
target.norm = None
results = importance_sampling(10000, target, proposal, lambda x: x)
print("Test mean:", results)  # it works as if norm == 1.0

# self.norm = 1/33.3, actually 1.0
target.norm = 1 / 33.3
results = importance_sampling(10000, target, proposal, lambda x: x)
print("Test mean:", results)  # fails, gets approximately [-0.0009, 0.0009, 0.0004]

print("")
# MultiGaussDenser,
# whose evaluate_density() returns 33.3x larger value
# self.norm = 1/33.3 in fact
target_denser = MultiGaussDenser(mean=test_mean, std=[1, 1, 1])
results = importance_sampling(10000, target_denser, proposal, lambda x: x)
print("Test mean:", results)  # it works as if norm == 1.0

# self.norm = None, actually 33.3
target_denser.norm = None
results = importance_sampling(10000, target_denser, proposal, lambda x: x)
print("Test mean:", results)  # it works as if norm == 1.0

# self.norm = 1.0, actually 33.3
target_denser.norm = 1.0
results = importance_sampling(10000, target_denser, proposal, lambda x: x)
print("Test mean:", results)  # fails, gets approximately [-33, 33, 17]

print("")
# torch.distributions.multivariate_normal.MultivariateNormal
target2 = Wrapper(MultivariateNormal(test_mean, torch.eye(3)))
proposal2 = Wrapper(MultivariateNormal(torch.zeros(3), torch.eye(3)))
results = importance_sampling(10000, target2, proposal2, lambda x: x)
print("Test mean:", results)
