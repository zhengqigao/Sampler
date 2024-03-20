import torch
from sampler.base import score_estimator
from sampler._common import Distribution
import torch.nn as nn

class MultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()

        # notice the difference between here and in test1_importance_sampling.py
        # here we make mean and std to be nn.Parameter. When users use score_estimator, they can backward through it,
        # to calculate a dloss/dmean and dloss/dstd.
        # Not necessarily make them to be nn.Parameter, as long as they require grads should be recognized by the torch.autograd.
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float32))
        self.std = nn.Parameter(torch.tensor(std, dtype=torch.float32))
        self.dim = len(mean)
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        if in_log:
            return -0.5 * (
                    torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
                    + torch.log(2 * torch.pi * self.std * self.std).sum()
            )
        else:
            return torch.exp(
                -0.5 * torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
            ) / (
                    torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim
                    * torch.prod(self.std)
            )


gaussinstance = MultiGauss([0], [1])
eval_func = lambda x: x ** 2  # essentially E_p(x) [x^2] = 1, as x ~ N(0, 1)
results = score_estimator(10000, gaussinstance, eval_func)

print("forward result:", results)

results.backward()

print("backward result")
