import sys
sys.path.append('..')
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

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
                    torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
                    + torch.log(2 * torch.pi * self.std * self.std).sum()
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

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
                torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
                + torch.log(2 * torch.pi * self.std * self.std).sum()
        ) + torch.log(torch.Tensor([33.3]))


# MultiGauss
# normal case
target = MultiGauss(mean=test_mean, std=[2, 1, 0.5])
proposal = MultiGauss(mean=[0, 0, 0], std=[2, 1, 0.5])
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
target_denser = MultiGaussDenser(mean=test_mean, std=[1, 2, 0.5])
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

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        ret = torch.exp(-(x ** 2) / 2) * torch.abs(torch.cos(x))
        ret = torch.sum(ret, dim=1)  # transpose row vector to column vector
        return torch.log(ret)



target1 = CustomDistribution1()
proposal1 = MultiGauss(mean=[0], std=[0.5])
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

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        ret = torch.cos(1 / x) ** 2
        ret[(x == 0) | (x < -1) | (x > 1)] = 0
        ret = torch.sum(ret, dim=1)  # transpose row vector to column vector
        return torch.log(ret)


target2 = CustomDistribution2()
proposal2 = MultiGauss(mean=[0], std=[5])
results, _ = importance_sampling(10000, target2, proposal2, lambda x: x)
print("Test mean:", results)
# -0.0031, which is close to 0

print("")

# torch.distributions.multivariate_normal.MultivariateNormal
# zhengqi: a random covariance matrix
cov = torch.randn(3, 3)
cov = torch.mm(cov, cov.t()) + torch.eye(3) * 0.05
target3 = Wrapper(MultivariateNormal(torch.Tensor(test_mean), cov))
proposal3 = Wrapper(MultivariateNormal(torch.zeros(3), torch.eye(3)))
results, _ = importance_sampling(10000, target3, proposal3, lambda x: x)
print("Test mean:", results)


# [-1, 1, .5]



class TensorizedMultiGauss(Distribution):
    def __init__(self, mean, std, device=torch.device("cpu")):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32).to(device)
        self.std = torch.tensor(std, dtype=torch.float32).to(device)
        self.dim = self.mean.shape
        self.device = device
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return (torch.randn((num_samples, *self.dim)).to(self.device) * self.std + self.mean)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
                torch.sum(((x - self.mean) / self.std) ** 2, dim=tuple(range(1, len(self.dim) + 1)))
                + torch.log(2 * torch.pi * self.std * self.std).sum()
        )


test_mean = torch.randn(3,2,2)  # one sample in this case is of shape (3,2,2)
# Please test the algorithm on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target = TensorizedMultiGauss(mean=test_mean, std=torch.abs(0.8*torch.ones(test_mean.shape)), device=device)
proposal = TensorizedMultiGauss(mean=0.5*test_mean, std=target.std, device=device)
proposal.mul_factor = 1.0
results, _ = importance_sampling(100000, target, proposal, lambda x: x)
print("estimated by IS", results)
print("true mean", test_mean)

# test if it is possible to feed in only a function
results, _ = importance_sampling(100000, target.evaluate_density, proposal, lambda x: x)
print("estimated by IS", results)
print("true mean", test_mean)