import torch
from ._common import Distribution
from torch.distributions import Distribution as TorchDistribution
import torch.distributions

## TODO: Make this wrapper class still have access to all other TorchDistribution methods.
class TDWrapper(Distribution):
    def __init__(self, distribution: TorchDistribution):
        super().__init__()
        self.distribution = distribution
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size([num_samples]))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x)


class LinearEnvelop1D(Distribution):
    def __init__(self, grid, derivate, eval_value):
        super().__init__()
        self.grid = torch.tensor(grid, dtype=torch.float32)
        self.derivate = torch.tensor(derivate, dtype=torch.float32)
        self.eval_value = eval_value.clone().detach()
        self.dim = self.grid.shape[1]
        self.mul_factor = 1.0
        assert self.grid.shape[0] == 2

    def sample(self, num_samples: int) -> torch.Tensor:
        intersect = ((self.eval_value[1] - self.derivate[1] * self.grid[1]) - (self.eval_value[0] - self.derivate[0] * self.grid[0]))/(self.derivate[0] - self.derivate[1])
        intersect_value = torch.exp(self.derivate[0] * intersect + self.eval_value[0] - self.derivate[0] * self.grid[0])
        A = intersect_value * (1 / self.derivate[0] - 1 / self.derivate[1])
        scale = A + (torch.exp(self.derivate[1]*((11*self.grid[1]-10*self.grid[0])-intersect) + ( self.eval_value[1] - self.derivate[1] * self.grid[1])))/self.derivate[1]
        # print("scale: {}".format(scale))
        samples = torch.rand(num_samples, self.dim) * scale
        upper_points = torch.gt(samples, intersect_value / self.derivate[0])
        lower_points = ~upper_points
        samples[upper_points] = (torch.log(self.derivate[1] * (samples[upper_points] - A)) - (
                                             self.eval_value[1] - self.derivate[1] * self.grid[1])) / self.derivate[1]
        samples[lower_points] = (torch.log(self.derivate[0] * samples[lower_points]) - (self.eval_value[0] - self.derivate[0] * self.grid[0]))/self.derivate[0]
        return samples

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        prob = torch.empty_like(x)
        mask = x < ((self.eval_value[1] - self.eval_value[0] + self.grid[0] * self.derivate[0] - self.grid[1] * self.derivate[1])/(self.derivate[0] - self.derivate[1]))
        prob[mask] = (self.derivate[0] * (x[mask] - self.grid[0, 0]) + self.eval_value[0]) + 0.01
        prob[~mask] = (self.derivate[1] * (x[~mask] - self.grid[1, 0]) + self.eval_value[1]) + 0.01
        return prob.view(-1)