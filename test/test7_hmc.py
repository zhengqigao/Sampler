from sampler.base import *
import torch
from sampler._common import Distribution, Condistribution
import matplotlib.pyplot as plt


class ConditionalMultiGauss(Condistribution):
    def __init__(self, std):
        super().__init__()
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(std)
        self.mul_factor = 1.0

    def sample(self, num_samples: int, y) -> torch.Tensor:
        # y has shape (m, d)
        # return shape (num_samples, m, d) with y as the mean
        assert len(y.shape) == 2 and y.shape[1] == self.dim
        return torch.randn((num_samples, y.shape[0], y.shape[1])) * self.std + y

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x is of shape (N,d), y is of shape (M,d)
        # return shape (N,M)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return -0.5 * (torch.sum(((x - y) / self.std) ** 2, dim=2) + torch.log(
            2 * torch.pi * self.std * self.std).sum())


class UnconditionalMultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(std)
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (torch.sum(((x - self.mean) / self.std) ** 2, dim=1) + torch.log(
            2 * torch.pi * self.std * self.std).sum())


samples = hamiltonian_monte_carlo(num_samples=10000,
                               target=UnconditionalMultiGauss([2, -2], [1, 1]),
                               step_size=0.1,
                               num_leapfrog=10,
                               initial=torch.rand(1, 2), # three different MC chains, sample independently
                               burn_in=0)


for i in range(samples.shape[1]):
    plt.figure()
    plt.scatter(samples[:,i, 0], samples[:, i, 1], s=1)
plt.show()

print(samples.shape)