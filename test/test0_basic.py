import numpy as np
import torch
from sampler.base import *
from sampler._common import Distribution
from typing import Optional
from torch.distributions.multivariate_normal import MultivariateNormal
from sampler.distribution import Wrapper

class MultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(mean)


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


try:
    instance1 = MultiGauss([1, 2], [1, 1]) # should raise an error,as we haven't defined the mul_factor or div_factor
    print(instance1.mul_factor)
except Exception as e:
    print("Error raised as expected, w/ error: ", e)


instance1.mul_factor = 2.0 # should be 1.0, but here we give mul_factor = 2.0 to test if div_factor can be returned correctly
print(instance1.div_factor) # should return 0.5
print(instance1.mul_factor)
instance1.mul_factor = 3.0 # say we find 2.0 is wrong, it should be mul_factor = 3.0
print(instance1.div_factor)
print(instance1.mul_factor)
instance1.div_factor = 3.0 # say we find actually it is div_factor = 3.0, not mul_factor = 3.0
print(instance1.div_factor)
print(instance1.mul_factor)

# remove the field mul_factor of instance1
instance2 = MultiGauss([1, 2], [1, 1])
instance2.div_factor = 2.0
print(instance2.mul_factor) # should return 0.5
print(instance2.div_factor)


