from sampler.base import *
from sampler._common import Distribution
from torch.distributions.multivariate_normal import MultivariateNormal
from sampler.distribution import Wrapper
from test_common_helper import MultiGauss, TensorizedMultiGauss


# MultiGaussDenser, returns 33.3x larger p.d.f. value
class MultiGaussDenser(Distribution):
    # same as MultiGauss, but evaluate_density() returns 33.3x larger value
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)
        self.std = std if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)
        self.dim = len(self.mean)
        self.mul_factor = 1 / 33.3

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (torch.sum(((x - self.mean) / self.std) ** 2, dim=1) + torch.log(2 * torch.pi * self.std * self.std).sum()) + torch.log(torch.Tensor([33.3]))


# CustomDstr1 is an unnormalized distribution
#       f(x) = exp(-x^2/2)*|cos(x)|
class CustomDstr1(Distribution):
    def __init__(self):
        super().__init__()
        self.mul_factor = None

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        ret = torch.exp(-(x**2) / 2) * torch.abs(torch.cos(x))
        ret = torch.sum(ret, dim=1)  # transpose row vector to column vector
        return torch.log(ret)


# CustomDstr2 is an unnormalized distribution
#       f(x) = cos^2(1/x) (-1 <= x < 0 or 0 < x <= 1)
#       f(0) = 0, specifically
class CustomDstr2(Distribution):
    def __init__(self):
        super().__init__()
        self.mul_factor = None

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        ret = torch.cos(1 / x) ** 2
        ret[(x == 0) | (x < -1) | (x > 1)] = 0
        ret = torch.sum(ret, dim=1)  # transpose row vector to column vector
        return torch.log(ret)


# WeirdDstr is a distribution with log_prob(x) = weirdness
# weirdness may be None, torch.nan, torch.inf
class WeirdDstr(Distribution):
    def __init__(self, weirdness):
        super().__init__()
        self.weirdness = weirdness
        self.mul_factor = None

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # fill an x-like tensor with weirdness
        return torch.tensor(self.weirdness, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)


# =============================================================================
# test mul_factor configuration
# =============================================================================
test_mean = [-1, 1, 0.5]
target = MultiGauss(mean=test_mean, std=[2, 1, 0.5])
proposal = MultiGauss(mean=[0, 0, 0], std=[2, 1, 0.5])
proposal.mul_factor = 1.0
results, _ = importance_sampling(10000, target, proposal, lambda x: x)
print(f"Test mean:{results}, true mean:{test_mean}")
# [-1, 1, .5]

target.mul_factor = None
results, _ = importance_sampling(10000, target, proposal, lambda x: x)
print(f"Test mean:{results}, true mean:{test_mean}")
# [-1, 1, .5], less accurate

target.mul_factor = 1 / 33.3  # wrong mul_factor
results, _ = importance_sampling(10000, target, proposal, lambda x: x)
print(f"Test mean:{results}, true mean:{test_mean}")
# 1/33.3*[-1, 1, .5], rescaled since mul_factor is wrong

print("")
# =============================================================================
# test support for custom defined distributions
# =============================================================================

# MultiGaussDenser, returns 33.3x larger p.d.f. value
target = MultiGaussDenser(mean=test_mean, std=[1, 2, 0.5])
target.mul_factor = 1 / 33.3  # correct
results, _ = importance_sampling(10000, target, proposal, lambda x: x)
print(f"Test mean:{results}, true mean:{test_mean}")
# [-1, 1, .5]

target.mul_factor = None
results, _ = importance_sampling(10000, target, proposal, lambda x: x)
print(f"Test mean:{results}, true mean:{test_mean}")
# [-1, 1, .5]

target.mul_factor = 1.0  # wrong
results, _ = importance_sampling(10000, target, proposal, lambda x: x)
print(f"Test mean:{results}, true mean:{test_mean}")
# [-1, 1, .5] * 33.3, rescaled since mul_factor is wrong

# CustomDstr1 is an unnormalized distribution:
#       f(x) = exp(-x^2/2)*|cos(x)|
target = CustomDstr1()
proposal = MultiGauss(mean=[0], std=[0.5])
results, _ = importance_sampling(10000, target, proposal, lambda x: x)
print(f"Test mean:{results}, true mean:{0}")
# output ≈ 0.0069, answer is 0

results, _ = importance_sampling(10000, target, proposal, lambda x: x**2)
print(f"Test mean:{results}, true mean:{1.1649}")
# output ≈ 0.6500, answer is 1.1649

# CustomDstr2 is an unnormalized distribution:
#       f(x) = cos^2(1/x) (-1 <= x < 0 or 0 < x <= 1)
#       f(0) = 0, specifically
target = CustomDstr2()
proposal = MultiGauss(mean=[0], std=[5])
results, _ = importance_sampling(10000, target, proposal, lambda x: x)
print(f"Test mean:{results}, true mean:{0}")
# output ≈ -0.0031, answer is 0

results, _ = importance_sampling(10000, target, proposal, lambda x: x**2)
print(f"Test mean:{results}, true mean:{0.1228}")
# output ≈ 0.1900, answer is 0.1228

# Weird distributions
target = MultivariateNormal(torch.zeros(3), torch.eye(3))
# which is unsupported without Wrapper
try:
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing unsupported Distribution object. Test mean:{results}")
except Exception as e:
    print(e)
    # rased TypeError: 'MultivariateNormal' object is not callable
target = WeirdDstr(weirdness=None)
try:
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing None log_prob. Test mean:{results}")
except Exception as e:
    print(e)
    # raised TypeError: must be real number, not NoneType
target = WeirdDstr(weirdness=torch.nan)
try:
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing NaN log_prob. Test mean:{results}")
except Exception as e:
    print(e)
    # outputs [nan]
target = WeirdDstr(weirdness=torch.inf)
try:
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing Infinity log_prob. Test mean:{results}")
except Exception as e:
    print(e)
    # outputs [nan]
target = WeirdDstr(weirdness=-torch.inf)
try:
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing -Infinity log_prob. Test mean:{results}")
except Exception as e:
    print(e)
    # outputs [nan]

# TODO: test wrong-dimension log_prob (really needed?)

print("")
# =============================================================================
# test if it is possible to feed in only a function
# =============================================================================

test_mean = [-1, 1, 0.5]
target = MultiGauss(mean=test_mean, std=[1, 1, 1])
proposal = MultiGauss(mean=[0, 0, 0], std=[5, 5, 5])
results, _ = importance_sampling(100000, target.log_prob, proposal, lambda x: x)
print(f"estimated by IS:{results}, true mean:{test_mean}")

# custom defined target function
target = lambda x: -0.5 * (torch.sum(((x - torch.tensor(test_mean, dtype=torch.float32)) / torch.tensor([2, 1, 0.5], dtype=torch.float32)) ** 2, dim=1) + torch.log(2 * torch.pi * torch.tensor([2, 1, 0.5], dtype=torch.float32) ** 2).sum())
proposal = Wrapper(MultivariateNormal(torch.zeros(3), 2 * torch.eye(3)))
results, _ = importance_sampling(10000, target, proposal, lambda x: x)
print(f"Test mean:{results}, true mean:{test_mean}")
# [-1, 1, .5]

# Fill None in x-like tensor
try:
    target = lambda x: torch.tensor(None, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing None target function. Test mean:{results}")
except Exception as e:
    print(e)
    # raised TypeError: must be real number, not NoneType

# Fill NaN in x-like tensor
try:
    target = lambda x: torch.tensor(torch.nan, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing NaN terget function. Test mean:{results}")
except Exception as e:
    print(e)
    # [nan, nan, nan]

# Fill Infinity in x-like tensor
try:
    target = lambda x: torch.tensor(torch.inf, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing Infinity target function. Test mean:{results}")
except Exception as e:
    print(e)
    # [nan, nan, nan]

# Fill -Infinity in x-like tensor
try:
    target = lambda x: torch.tensor(-torch.inf, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing -Infinity target function. Test mean:{results}")
except Exception as e:
    print(e)
    # [nan, nan, nan]

# TODO: test wrong-dimension function

print("")
# =============================================================================
# wrapper
# =============================================================================

# torch.distributions.multivariate_normal.MultivariateNormal
# zhengqi: a random covariance matrix
cov = torch.randn(3, 3)
cov = torch.mm(cov, cov.t()) + torch.eye(3) * 0.05
target = Wrapper(MultivariateNormal(torch.Tensor(test_mean), cov))
proposal = Wrapper(MultivariateNormal(torch.zeros(3), torch.eye(3)))
results, _ = importance_sampling(10000, target, proposal, lambda x: x)
print(f"Test mean:{results}, true mean:{test_mean}")
# [-1, 1, .5]

# TODO: test wrapping None/NaN/Inf/wrong-dimension tensor

print("")
# =============================================================================
# sampling tensors
# =============================================================================

test_mean = torch.randn(3, 2, 2)  # one sample in this case is of shape (3,2,2)
# Please test the algorithm on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target = TensorizedMultiGauss(mean=test_mean, std=torch.abs(0.8 * torch.ones(test_mean.shape)), device=device)
proposal = TensorizedMultiGauss(mean=0.5 * test_mean, std=target.std, device=device)
proposal.mul_factor = 1.0
results, _ = importance_sampling(100000, target, proposal, lambda x: x)
print("estimated by IS", results)
print("true mean", test_mean)

# TODO: test None/NaN/Inf/wrong-dimension tensorized target/proposal
