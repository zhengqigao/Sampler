from sampler.base import *
from sampler._common import Distribution, _BaseDistribution



class MultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)
        self.std = std if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)
        self.dim = len(mean)

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
                torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
                + torch.log(2 * torch.pi * self.std * self.std).sum()
            )



instance1 = MultiGauss([1, 2], [1, 1])
instance1.sample(10)

try:
    instance1.mul_factor = 0  # should raise an error, as mul_factor should be positive
    print(instance1.mul_factor)
except Exception as e:
    print("Error raised as expected, w/ error: ", e)

try:
    # should raise an error,as we haven't defined the mul_factor or div_factor
    print(instance1.mul_factor)
except Exception as e:
    print("Error raised as expected, w/ error: ", e)

# should be 1.0, but here we give mul_factor = 2.0 to test if div_factor can be returned correctly
instance1.mul_factor = 2.0
print(instance1.mul_factor, instance1.div_factor)  # should return 2.0 0.5

# say we find 2.0 is wrong, it should be mul_factor = 3.0
instance1.mul_factor = 3.0
print(instance1.mul_factor, instance1.div_factor)  # 3.0 0.3333333333333333

# say we find actually it is div_factor = 3.0, not mul_factor = 3.0
instance1.div_factor = 3.0
print(instance1.mul_factor, instance1.div_factor)  # 0.3333333333333333 3.0

# remove mul_factor
instance1.mul_factor = None
print(instance1.mul_factor, instance1.div_factor)  # None None

# remove the field mul_factor of instance1
instance2 = MultiGauss([1, 2], [1, 1])
instance2.div_factor = 2.0
print(instance2.mul_factor, instance2.div_factor)  # should return 0.5 2.0

s = instance2.sample(10)
print(s)