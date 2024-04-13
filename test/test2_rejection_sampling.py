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
results, info = rejection_sampling(10000, target, proposal, k=1000.0)

print(results.shape)
print(torch.mean(results, dim=0))
print(info['rejection_rate'])
