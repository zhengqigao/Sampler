import torch
from ._common import Distribution
from torch.distributions import Distribution as TorchDistribution

class Wrapper(Distribution):
    def __init__(self, distribution: TorchDistribution):
        super().__init__()
        self.distribution = distribution
        self.norm = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    def evaluate_density(self, x: torch.Tensor, in_log: bool = False) -> torch.Tensor:
        if in_log:
            return self.distribution.log_prob(x)
        else:
            return torch.exp(self.distribution.log_prob(x))

