from sampler.base import MH_sampling
import torch
from sampler._common import Wrapper, Distribution
import matplotlib.pyplot as plt




class ConditionalMultiGauss(Distribution):
    def __init__(self, std):
        super().__init__()
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(std)
        self.const = 1.0

    def sample(self, num_samples: int, y) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + y

    def evaluate_density(self, x: torch.Tensor, y: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        if in_log:
            return -0.5 * (torch.sum(((x - y) / self.std) ** 2, dim=1) + torch.log(
                torch.tensor(2 * torch.pi)) * self.dim)
        else:
            return torch.exp(-0.5 * torch.sum(((x - y) / self.std) ** 2, dim=1)) / (
                    torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim * torch.prod(self.std))

class UnconditionalMultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(std)
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


gauss1 = ConditionalMultiGauss(std = [1, 1])
gauss2 = UnconditionalMultiGauss([-1,1], [1, 1])
results, info = MH_sampling(10000, gauss2, gauss1, torch.zeros((1,2)))

plt.figure()
plt.scatter(results[:, 0], results[:, 1], s=1)
plt.show()

print(f"info['acceptance_rate'] = {info['acceptance_rate']}")