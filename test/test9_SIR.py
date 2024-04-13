import torch

from sampler.base import *
from sampler._common import Distribution

test_mean = [-1, 1, 0.5]


class MultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)
        self.std = std if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)
        self.dim = len(mean)
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (torch.sum(((x - self.mean) / self.std) ** 2, dim=1) + torch.log(
            2 * torch.pi * self.std * self.std).sum())


target = MultiGauss(mean=test_mean, std=[1, 1, 1])
target.mul_factor = None

proposal = MultiGauss(mean=[0, 0, 0], std=[1, 1, 1])
expectation, test = importance_sampling(100000, target, proposal,eval_func=lambda x: x, resample_ratio=0.5)
print("Test mean:", expectation)
print(torch.mean(test, dim=0))
print(torch.std(test, dim=0))


# MultiGauss
# normal case
target = MultiGauss(mean=test_mean, std=[1, 1, 1])
proposal = MultiGauss(mean=[0, 0, 0], std=[1, 1, 1])
proposal.mul_factor = 1.0
results = importance_sampling(100000, target, proposal, lambda x: x)
# it works, [-1, 1, .5]
