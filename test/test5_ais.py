from sampler.base import *
import torch
from sampler._common import Distribution, Condistribution


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
            return -0.5 * (torch.sum(((x - y) / self.std) ** 2, dim=2) + torch.log(
                2 * torch.pi * self.std * self.std).sum())
        else:
            return torch.exp(-0.5 * torch.sum(((x - y) / self.std) ** 2, dim=2)) / (
                    torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim * torch.prod(self.std * self.std))


class UnconditionalMultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(std)
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        if in_log:
            return -0.5 * (torch.sum(((x - self.mean) / self.std) ** 2, dim=1) + torch.log(
                2 * torch.pi * self.std * self.std).sum())
        else:
            return torch.exp(-0.5 * torch.sum(((x - self.mean) / self.std) ** 2, dim=1)) / (
                    torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim * torch.prod(self.std * self.std))


results = annealed_importance_sampling(num_samples=50000,
                                       target=UnconditionalMultiGauss([2, -2], [1, 1]),
                                       base=UnconditionalMultiGauss([0, 0], [1, 1]),
                                       transit=ConditionalMultiGauss(std=[1, 1]),
                                       eval_func=lambda x: x,
                                       beta=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                       anneal_log_criterion=lambda logpt, logpb, beta: beta * logpt + (
                                                   1 - beta) * logpb)

print(results)
