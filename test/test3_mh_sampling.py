from sampler.base import *
import torch
from sampler._common import Distribution, Condistribution
import matplotlib.pyplot as plt
from sampler.distribution import Wrapper



class ConditionalMultiGauss(Condistribution):
    def __init__(self, std):
        super().__init__()
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(std)
        self.const = 1.0

    def sample(self, num_samples: int, y) -> torch.Tensor:
        # y has shape (m, d)
        # return shape (num_samples, m, d) with y as the mean
        assert len(y.shape) == 2 and y.shape[1] == self.dim
        return torch.randn((num_samples, y.shape[0], y.shape[1])) * self.std + y

    def evaluate_density(self, x: torch.Tensor, y: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        # x is of shape (N,d), y is of shape (M,d)
        # return shape (N,M)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        if in_log:
            return -0.5 * (torch.sum(((x - y) / self.std) ** 2, dim=2) + torch.log(2 * torch.pi * self.std * self.std).sum())
        else:
            return torch.exp(-0.5 * torch.sum(((x - y) / self.std) ** 2, dim=2)) / (
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
            return -0.5 * (torch.sum(((x - self.mean) / self.std) ** 2, dim=1) + torch.log(2 * torch.pi * self.std * self.std).sum())
        else:
            return torch.exp(-0.5 * torch.sum(((x - self.mean) / self.std) ** 2, dim=1)) / (
                    torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim * torch.prod(self.std))


gauss1 = ConditionalMultiGauss(std = [1, 1])
gauss2 = UnconditionalMultiGauss([-1,1], [1, 1])
results, info = mh_sampling(10000, gauss2, gauss1, torch.zeros((3, 2))) # 3 different MC chains, each grown by MH independently

for batch_index in range(results.shape[1]):
    plt.figure()
    plt.scatter(results[:, batch_index, 0], results[:, batch_index, 1], s=1)
plt.show()

print(f"info['acceptance_rate'] = {info['acceptance_rate']}")