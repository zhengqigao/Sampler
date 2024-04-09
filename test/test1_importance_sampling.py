from sampler.base import *
from sampler._common import Distribution
from torch.distributions.multivariate_normal import MultivariateNormal
from sampler.distribution import Wrapper

test_mean = [-1, 1, 0.5]


class MultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(self.mean)
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
                * torch.prod(self.std  * self.std)
            )


class MultiGaussDenser(Distribution):
    # same as MultiGauss, but evaluate_density() returns 33.3x larger value
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(self.mean)
        self.mul_factor = 1 / 33.3

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        if in_log:
            return -0.5 * (
                torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
                + torch.log(2 * torch.pi * self.std * self.std).sum()
            ) + torch.log(torch.Tensor([33.3]))
        else:
            return (
                torch.exp(-0.5 * torch.sum(((x - self.mean) / self.std) ** 2, dim=1))
                / (
                    torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim
                    * torch.prod(self.std)
                )
                * 33.3
            )


# MultiGauss
# normal case
target = MultiGauss(mean=test_mean, std=[1, 1, 1])
proposal = MultiGauss(mean=[0, 0, 0], std=[1, 1, 1])
proposal.mul_factor = 1.0
results, _ = importance_sampling(10000, target, proposal, lambda x: x)
print("Test mean:", results)
# it works, [-1, 1, .5]

# self.mul_factor = None, actually 1.0
target.mul_factor = None
results, _ = importance_sampling(10000, target, proposal, lambda x: x)
print("Test mean:", results)
# [-1, 1, .5]


# self.mul_factor = 1/33.3, actually 1.0
target.mul_factor = 1 / 33.3
results, _ = importance_sampling(10000, target, proposal, lambda x: x)
print("Test mean:", results)
# [-1, 1, .5]


print("")


# MultiGaussDenser,
# whose evaluate_density() returns 33.3x larger value
# self.mul_factor = 1/33.3 in fact
target_denser = MultiGaussDenser(mean=test_mean, std=[1, 1, 1])
target_denser.mul_factor = 1 / 33.3
results, _ = importance_sampling(10000, target_denser, proposal, lambda x: x)
print("Test mean:", results)
# also works, [-1, 1, .5]


# self.mul_factor = None, actually 1/33.3
target_denser.mul_factor = None
results, _ = importance_sampling(10000, target_denser, proposal, lambda x: x)
print("Test mean:", results)
# [-1, 1, .5]


# self.mul_factor = 1.0, actually 1/33.3
target_denser.mul_factor = 1.0
results, _ = importance_sampling(10000, target_denser, proposal, lambda x: x)
print("Test mean:", results)
# rescaled, [-1, 1, .5] * 33.3


print("")


# CustomDistribution1 is an unnormalized distribution
# f(x) = exp(-x^2/2)*|cos(x)|
class CustomDistribution1(Distribution):
    def __init__(self):
        super().__init__()
        self.mul_factor = None

    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        ret = torch.exp(-(x**2) / 2) * torch.abs(torch.cos(x))
        ret = torch.sum(ret, dim=1)  # transpose row vector to column vector
        if in_log:
            return torch.log(ret)
        else:
            return ret


target1 = CustomDistribution1()
proposal1 = MultiGauss(mean=[0], std=[1])
results, _ = importance_sampling(10000, target1, proposal1, lambda x: x)
print("Test mean:", results)
# 0.0069, which is close to 0


# CustomDistribution2 is an unnormalized distribution
# f(x) = cos^2(1/x) (-1 <= x < 0 or 0 < x <= 1)
# f(0) = 0, specifically
class CustomDistribution2(Distribution):
    def __init__(self):
        super().__init__()
        self.mul_factor = None

    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        ret = torch.cos(1 / x) ** 2
        ret[(x == 0) | (x < -1) | (x > 1)] = 0
        ret = torch.sum(ret, dim=1)  # transpose row vector to column vector
        if in_log:
            return torch.log(ret)
        else:
            return ret

target2 = CustomDistribution2()
proposal2 = MultiGauss(mean=[0], std=[5])
results, _ = importance_sampling(10000, target2, proposal2, lambda x: x)
print("Test mean:", results)
# -0.0031, which is close to 0

print("")


# torch.distributions.multivariate_normal.MultivariateNormal
target3 = Wrapper(MultivariateNormal(torch.Tensor(test_mean), torch.eye(3)))
proposal3 = Wrapper(MultivariateNormal(torch.zeros(3), torch.eye(3)))
results, _ = importance_sampling(10000, target3, proposal3, lambda x: x)
print("Test mean:", results)
# [-1, 1, .5]
