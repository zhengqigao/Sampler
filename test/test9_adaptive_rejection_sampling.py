import torch

from sampler.base import *
from sampler._common import Distribution


class MultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(mean)
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (torch.sum(((x - self.mean) / self.std) ** 2, dim=1) + torch.log(
            2 * torch.pi * self.std * self.std).sum())


test_mean = [1]
test_std = [1]
lower = -1
upper = 3
target = MultiGauss(mean=test_mean, std=test_std)
target.mul_factor = None

results, info = adaptive_rejection_sampling(10000, target, lower, upper)

print(results.shape)
print(torch.mean(results, dim=0))
print(info['rejection_rate'])


