import numpy as np
import torch
from sampler.base import importance_sampling
from sampler._common import Distribution
from typing import Optional
from torch.distributions.multivariate_normal import MultivariateNormal
from sampler._common import Wrapper

test_mean = torch.Tensor([-1, 1, 0.5])


class MultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(mean)
        self.const = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        if in_log:
            return -0.5 * (torch.sum(((x - self.mean) / self.std) ** 2, dim=1) + torch.log(
                torch.tensor(2 * torch.pi)) * self.dim)
        else:
            return torch.exp(-0.5 * torch.sum(((x - self.mean) / self.std) ** 2, dim=1)) / (
                    torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim * torch.prod(self.std))


target = MultiGauss(mean=test_mean, std=[1, 1, 1])
proposal = MultiGauss(mean=[0, 0, 0], std=[1, 1, 1])
results = importance_sampling(10000, target, proposal, lambda x: x)
print("Test mean:", results)

target2 = Wrapper(MultivariateNormal(test_mean, torch.eye(3)))
proposal2 = Wrapper(MultivariateNormal(torch.zeros(3), torch.eye(3)))
results = importance_sampling(10000, target2, proposal2, lambda x: x)
print("Test mean:", results)


