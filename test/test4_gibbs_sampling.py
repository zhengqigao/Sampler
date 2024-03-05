from sampler.base import *
import torch
from sampler._common import Wrapper, Distribution, Condistribution
import matplotlib.pyplot as plt

class ConditionalMultiGauss(Condistribution):
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


## TODO: use an example of joint Gaussian, so that we know all conditional distributions are Gaussian. Test Gibbs sampling on this example.

