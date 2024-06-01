from typing import Union

import torch
from ._common import Distribution, checker
from torch.distributions import Distribution as TorchDistribution
import torch.distributions

## TODO: Make this wrapper class still have access to all other TorchDistribution methods.
class TDWrapper(Distribution):
    def __init__(self, distribution: TorchDistribution):
        super().__init__()
        self.distribution = distribution
        self.mul_factor = 1.0

    @checker
    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size([num_samples]))

    @checker
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x)


class LinearEnvelop1D(Distribution):
    def __init__(self, grid, derivative, eval_value):
        super().__init__()
        self.grid = grid if isinstance(grid, torch.Tensor) else torch.tensor(grid, dtype=torch.float32)
        self.derivative = derivative.clone().detach().requires_grad_(False)
        self.eval_value = eval_value.clone().detach()
        self.dim = self.grid.shape[1]
        self.mul_factor = 1.0

        partition = self.grid.shape[0]
        intersect = []
        intersect_value = []
        t = []
        A = []
        A_sum = []
        for j in range(partition - 1):
            intersect_j = ((eval_value[j+1]-derivative[j+1]*grid[j+1]) - (eval_value[j]-derivative[j]*grid[j])) / (derivative[j]-derivative[j+1])
            intersect.append(intersect_j)
            t_j = eval_value[j]-derivative[j]*grid[j]
            t.append(t_j)
            intersect_value_j = torch.exp(derivative[j] * intersect_j + t_j)
            intersect_value.append(intersect_value_j)
            if j == 0:
                A.append(None)
                A_sum.append(0)
            else:
                A_j = intersect_value[j-1] * (1 / self.derivative[j-1] - 1 / self.derivative[j])
                A.append(A_j)
                A_sum.append(A_j+A_sum[j-1])

        j = partition-1
        t_j = eval_value[j] - derivative[j] * grid[j]
        t.append(t_j)
        A_j = intersect_value[j - 1] * (1 / self.derivative[j - 1] - 1 / self.derivative[j])
        A.append(A_j)
        A_sum.append(A_j + A_sum[j - 1])

        self.intersect = intersect
        self.intersect_value = intersect_value
        self.t = t
        self.A = A
        self.A_sum = A_sum

    def sample(self, num_samples: int) -> torch.Tensor:
        scale = self.A_sum[-1] + (torch.exp(self.derivative[-1] * (100 * self.grid[-1] - 99 * self.grid[0] + self.intersect[-1]) + (self.eval_value[-1] - self.derivative[-1] * self.grid[-1])
                                       )) / self.derivative[-1]

        samples_ori = torch.rand(num_samples, self.dim) * scale
        samples = torch.zeros_like(samples_ori)

        upper_points = torch.gt(samples_ori, self.intersect_value[0] / self.derivative[0])
        lower_points = ~upper_points

        samples[lower_points] = (torch.log(self.derivative[0] * samples_ori[lower_points]) -
                                 (self.eval_value[0] - self.derivative[0] * self.grid[0])) / self.derivative[0]

        upper_points = torch.gt(samples_ori, self.intersect_value[-1] / self.derivative[-2] + self.A_sum[-2])

        samples[upper_points] = (torch.log(self.derivative[-1] * (samples_ori[upper_points] - self.A_sum[-1])) - self.t[-1]
                             ) / self.derivative[-1]

        for j in range(1, len(self.intersect)):
            larger = ~torch.lt(samples_ori, self.intersect_value[j-1] / self.derivative[j-1] + self.A_sum[j-1])
            smaller = torch.lt(samples_ori, self.intersect_value[j] / self.derivative[j] + self.A_sum[j])
            mask = torch.logical_and(larger, smaller)
            samples[mask] = (torch.log(self.derivative[j] * (samples_ori[mask] - self.A_sum[j])) - self.t[j]
                             ) / self.derivative[j]

        return samples

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        prob = torch.empty_like(x)
        mask = x < self.intersect[0]
        prob[mask] = (self.derivative[0] * (x[mask] - self.grid[0, 0]) + self.eval_value[0]) + 0.01

        for j in range(1, len(self.intersect)):
            larger = ~torch.lt(x, self.intersect[j-1])
            smaller = torch.lt(x, self.intersect[j])
            mask = torch.logical_and(larger, smaller)
            prob[mask] = (self.derivative[j] * (x[mask] - self.grid[j]) + self.eval_value[j]) + 0.01
        j = len(self.intersect)
        mask = x >= self.intersect[j-1]
        prob[mask] = (self.derivative[j] * (x[mask] - self.grid[j]) + self.eval_value[j]) + 0.01
        return prob.view(-1)
