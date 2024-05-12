import torch
import sys
sys.path.append('../')
from sampler._common import Distribution, checker


class UnconditionalMultiGauss1(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)
        self.std = std if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)
        self.dim = len(self.mean)
        self.mul_factor = 1.0

    @checker
    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    @checker
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
                torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
                + torch.log(2 * torch.pi * self.std * self.std).sum()
        )

class UnconditionalMultiGauss2(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)
        self.std = std if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)

        self.mean = torch.nn.Parameter(self.mean)
        self.std = torch.nn.Parameter(self.std)

        self.dim = len(self.mean)
        self.mul_factor = 1.0

    @checker
    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    @checker
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
                torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
                + torch.log(2 * torch.pi * self.std * self.std).sum()
        )



test_p = UnconditionalMultiGauss1(mean=[0.0, 0.0], std=[1.0, 1.0])
test_p.sample(10)
test_p.log_prob(torch.randn(10,2))

test_p.to('cpu')
test_p.sample(10)
test_p.log_prob(torch.randn(10,2))

test_p.to('cuda')
test_p.sample(10)
test_p.log_prob(torch.randn(10,2))