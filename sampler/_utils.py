from typing import TypeVar, Callable, Union, Optional
import torch
import torch.nn as nn
from ._common import Func, Distribution

## TODO: define an alias decorator, so maybe we can call ais or annealed_importance_sampling for the same function?
def _alias(*aliases):
    def decorator(func):
        for alias in aliases:
            globals()[alias] = func
        return func
    return decorator

def _get_params(module: Union[torch.nn.Module, torch.nn.DataParallel]):
    if isinstance(module, torch.nn.DataParallel):
        return _get_params(module.module)
    else:
        return module.parameters()


class _ModuleWrapper(nn.Module):
    def __init__(self, func: Func):
        super(_ModuleWrapper, self).__init__()
        self.func = func
    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class LinearEnvelop1D(Distribution):
    def __init__(self, grid, derivate, eval_value):
        super().__init__()
        self.grid = torch.tensor(grid, dtype=torch.float32)
        self.derivate = torch.tensor(derivate, dtype=torch.float32)
        self.eval_value = eval_value.clone().detach()
        self.dim = self.grid.shape[1]
        assert self.grid.shape[0] == 2

    def sample(self, num_samples: int) -> torch.Tensor:
        intersect = ((self.eval_value[1] - self.derivate[1] * self.grid[1]) - (self.eval_value[0] - self.derivate[0] * self.grid[0]))/(self.derivate[0] - self.derivate[1])
        samples = torch.rand(num_samples, self.dim)
        intersect_value = torch.exp(self.derivate[0] * intersect + self.eval_value[0] - self.derivate[0] * self.grid[0])
        upper_points = torch.gt(samples, intersect_value)
        lower_points = ~upper_points
        samples[upper_points] = (torch.log(self.derivate[1] * (samples[upper_points] - torch.exp(self.derivate[0] * intersect + self.eval_value[0] - self.derivate[0] * self.grid[0])/self.derivate[0]))-(self.eval_value[0] - self.derivate[0] * self.grid[0])) / self.derivate[1] + intersect
        samples[lower_points] = (torch.log(self.derivate[0] * samples[lower_points]) - (self.eval_value[0] - self.derivate[0] * self.grid[0]))/self.derivate[0]
        return samples

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if x < (self.eval_value[1] - self.eval_value[0] + self.grid[0] * self.derivate[0] - self.grid[1] * self.derivate[1])/(self.derivate[0] - self.derivate[1]):
            return self.derivate[0] * (x - self.grid[0]) + self.eval_value[0]
        else:
            return self.derivate[1] * (x - self.grid[1]) + self.eval_value[1]