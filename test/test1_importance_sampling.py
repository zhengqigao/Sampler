from sampler.base import *
from sampler._common import Distribution
from torch.distributions.multivariate_normal import MultivariateNormal
from sampler.distribution import TDWrapper
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

# None div_factor
target.div_factor = None
results, _ = importance_sampling(100, target, proposal, lambda x: x)
print(f"Test mean:{results}, true mean:{test_mean}")

# Other weird mul_factor and div_factor
try:
    target.mul_factor = torch.nan
    results, _ = importance_sampling(100, target, proposal, lambda x: x)
    print(f"Test mean:{results}, true mean:{test_mean}")
except Exception as e:
    print(e)
    # raised ValueError: The mul_factor must be a positive scalar, but got nan.
try:
    target.div_factor = torch.nan
    results, _ = importance_sampling(100, target, proposal, lambda x: x)
    print(f"Test mean:{results}, true mean:{test_mean}")
except Exception as e:
    print(e)
    # raised ValueError: The div_factor must be a positive scalar, but got nan.
try:
    target.mul_factor = torch.inf
    results, _ = importance_sampling(100, target, proposal, lambda x: x)
    print(f"Test mean:{results}, true mean:{test_mean}")
except Exception as e:
    print(e)
    # raised ValueError: The mul_factor must be a positive scalar, but got inf.
try:
    target.div_factor = torch.inf
    results, _ = importance_sampling(100, target, proposal, lambda x: x)
    print(f"Test mean:{results}, true mean:{test_mean}")
except Exception as e:
    print(e)
    # raised ValueError: The div_factor must be a positive scalar, but got inf.
try:
    target.mul_factor = -torch.inf
    results, _ = importance_sampling(100, target, proposal, lambda x: x)
    print(f"Test mean:{results}, true mean:{test_mean}")
except Exception as e:
    print(e)
    # raised ValueError: The mul_factor must be a positive scalar, but got -inf.
try:
    target.div_factor = -torch.inf
    results, _ = importance_sampling(100, target, proposal, lambda x: x)
    print(f"Test mean:{results}, true mean:{test_mean}")
except Exception as e:
    print(e)
    # raised ValueError: The div_factor must be a positive scalar, but got -inf.

print("")
# =============================================================================
# test support for custom defined distributions
# =============================================================================
test_mean = [-1, 1, 0.5]
# MultiGaussDenser, returns 33.3x larger p.d.f. value
target = MultiGaussDenser(mean=test_mean, std=[1, 1, 1])
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
try:
    target = MultivariateNormal(torch.zeros(3), torch.eye(3))
    # which is unsupported without TDWrapper
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing unsupported Distribution object. Test mean:{results}")
except Exception as e:
    print(e)
    # rased TypeError: 'MultivariateNormal' object is not callable
try:
    target = Distribution()
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing unsupported Distribution object. Test mean:{results}")
except Exception as e:
    print(e)
    # raised NotImplementedError (printed an empty line)
try:
    target = MultiGauss(mean=test_mean, std=[1, 1, 1])
    proposal = MultiGauss(mean=[0, 0, 0], std=[5, 5, 5])
    # overwrite log_prob with x-like tensor filled with None
    proposal.log_prob = lambda x: torch.tensor(None, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing None proposal.log_prob. Test mean:{results}")
except Exception as e:
    print(e)
    # raised TypeError: must be real number, not NoneType
    # This message is originally raised by PyTorch when trying to put None into a tensor.
    # As for WHAT must not be NoneType, the user should inspect the CALL STACK.
try:
    target = MultiGauss(mean=test_mean, std=[1, 1, 1])
    # overwrite log_prob with x-like tensor filled with None
    target.log_prob = lambda x: torch.tensor(None, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    proposal = MultiGauss(mean=[0, 0, 0], std=[5, 5, 5])
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing None target.log_prob. Test mean:{results}")
except Exception as e:
    print(e)
    # raised TypeError: must be real number, not NoneType
    # This message is originally raised by PyTorch when trying to put None into a tensor.
    # As for WHAT must not be NoneType, the user should inspect the CALL STACK.

try:
    target = MultiGauss(mean=test_mean, std=[1, 1, 1])
    # overwrite log_prob with x-like tensor filled with NaN
    target.log_prob = lambda x: torch.tensor(torch.nan, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    proposal = MultiGauss(mean=[0, 0, 0], std=[5, 5, 5])
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing NaN target.log_prob. Test mean:{results}")
except Exception as e:
    print(e)
    # warning: target log_prob returns NaN.
    # [nan,nan,nan]
try:
    target = MultiGauss(mean=test_mean, std=[1, 1, 1])
    # overwrite log_prob with x-like tensor filled with Inf
    target.log_prob = lambda x: torch.tensor(torch.inf, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    proposal = MultiGauss(mean=[0, 0, 0], std=[5, 5, 5])
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing Inf target.log_prob. Test mean:{results}")
except Exception as e:
    print(e)
    # [nan,nan,nan]
try:
    target = MultiGauss(mean=test_mean, std=[1, 1, 1])
    # overwrite log_prob with x-like tensor filled with -Inf
    target.log_prob = lambda x: torch.tensor(-torch.inf, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    proposal = MultiGauss(mean=[0, 0, 0], std=[5, 5, 5])
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing -Inf target.log_prob. Test mean:{results}")
    target.mul_factor = None
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing -Inf target.log_prob. Test mean:{results}")
except Exception as e:
    print(e)
    # [0,0,0] and [nan,nan,nan]

print("")
# =============================================================================
# test if it is possible to feed in only a function
# =============================================================================

test_mean = [-1, 1, 0.5]
target = MultiGauss(mean=test_mean, std=[1, 1, 1])
proposal = MultiGauss(mean=[0, 0, 0], std=[5, 5, 5])
results, _ = importance_sampling(100000, target.log_prob, proposal, lambda x: x)
print(f"Test mean:{results}, true mean:{test_mean}")

# custom defined target function
target = lambda x: -0.5 * (torch.sum(((x - torch.tensor(test_mean, dtype=torch.float32)) / torch.tensor([2, 1, 0.5], dtype=torch.float32)) ** 2, dim=1) + torch.log(2 * torch.pi * torch.tensor([2, 1, 0.5], dtype=torch.float32) ** 2).sum())
proposal = TDWrapper(MultivariateNormal(torch.zeros(3), 2 * torch.eye(3)))
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
try:
    # Fill NaN in x-like tensor
    target = lambda x: torch.tensor(torch.nan, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing NaN terget function. Test mean:{results}")
except Exception as e:
    print(e)
    # warning: target log_prob returns NaN.
    # [nan, nan, nan]
try:
    # Fill Inf in x-like tensor
    target = lambda x: torch.tensor(torch.inf, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing Inf target function. Test mean:{results}")
except Exception as e:
    print(e)
    # [nan, nan, nan]
try:
    # Fill -Inf in x-like tensor
    target = lambda x: torch.tensor(-torch.inf, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Testing -Inf target function. Test mean:{results}")
except Exception as e:
    print(e)
    # [nan, nan, nan]
    # kaiwen: this behavior is different from obj with mul_factor (i.e., [0,0,0]). So a warning for this case is added.

# wrong-dimension target function
try:
    target = MultiGauss(mean=test_mean, std=[1, 1, 1])
    proposal = MultiGauss(mean=[0, 0, 0], std=[5, 5, 5])
    # target function ignores the last sample
    results, _ = importance_sampling(100000, lambda x: target.log_prob(x)[:-1], proposal, lambda x: x)
    print(f"Test mean:{results}, true mean:{test_mean}")
except Exception as e:
    print(e)
    # raiesd RuntimeError: The size of tensor a (99999) must match the size of tensor b (100000)
    #                      at non-singleton dimension 0
try:
    target = MultiGauss(mean=test_mean, std=[1, 1, 1])
    proposal = MultiGauss(mean=[0, 0, 0], std=[5, 5, 5])
    # target function ignores the last sample
    results, _ = importance_sampling(100000, lambda x: target.log_prob(x)[:, :-1], proposal, lambda x: x)
    print(f"Test mean:{results}, true mean:{test_mean}")
except Exception as e:
    print(e)
    # raiesd IndexError: too many indices for tensor of dimension 1


print("")
# =============================================================================
# TDwrapper
# =============================================================================

# torch.distributions.multivariate_normal.MultivariateNormal
# zhengqi: a random covariance matrix
test_mean = [-1, 1, 0.5]
cov = torch.randn(3, 3)
cov = torch.mm(cov, cov.t()) + torch.eye(3) * 0.05
target = TDWrapper(MultivariateNormal(torch.Tensor(test_mean), cov))
proposal = TDWrapper(MultivariateNormal(torch.zeros(3), torch.eye(3)))
results, _ = importance_sampling(10000, target, proposal, lambda x: x)
print(f"Test mean:{results}, true mean:{test_mean}")
# [-1, 1, .5]

# kaiwen: not rewriting these error message
try:
    target = TDWrapper(MultivariateNormal(torch.Tensor(test_mean[:1]), cov[:1, :1]))
    proposal = TDWrapper(MultivariateNormal(torch.zeros(3), torch.eye(3)))
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Test mean:{results}, true mean:{test_mean[:1]}")
except Exception as e:
    print(e)
    # raised ValueError: The right-most size of value must match event_shape:
    #                    torch.Size([10000, 3]) vs torch.Size([1]).
try:
    target = TDWrapper(MultivariateNormal(torch.Tensor(test_mean), cov))
    proposal = TDWrapper(MultivariateNormal(torch.zeros(4), torch.eye(4)))
    results, _ = importance_sampling(10000, target, proposal, lambda x: x)
    print(f"Test mean:{results}, true mean:{test_mean[:1]}")
except Exception as e:
    print(e)
    # raised ValueError: The right-most size of value must match event_shape:
    #                    torch.Size([10000, 4]) vs torch.Size([3]).

print("")
# =============================================================================
# sampling tensors
# =============================================================================

test_mean = torch.randn(3, 2, 2)  # one sample in this case is of shape (3,2,2)
# Please test the algorithm on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
target = TensorizedMultiGauss(mean=test_mean, std=torch.abs(0.8 * torch.ones(test_mean.shape)), device=device)
proposal = TensorizedMultiGauss(mean=0.5 * test_mean, std=target.std, device=device)
proposal.mul_factor = 1.0
results, _ = importance_sampling(100000, target, proposal, lambda x: x)
print("estimated by IS", results)
print("true mean", test_mean)

# None tensor
try:
    target = TensorizedMultiGauss(mean=None, std=torch.abs(0.8 * torch.ones(test_mean.shape)), device=device)
    results, _ = importance_sampling(100000, target, proposal, lambda x: x)
    print(f"Testing None tensor. Test mean:{results}")
except Exception as e:
    print(e)
    # raised TypeError: must be real number, not NoneType
# NaN tensor
try:
    # fill an x-like tensor with NaN
    target = lambda x: torch.tensor(torch.nan, dtype=torch.float32).to(device) * torch.ones(x.shape[0], dtype=torch.float32).to(device)
    results, _ = importance_sampling(100000, target, proposal, lambda x: x)
    print(f"Testing NaN tensor. Test mean:{results}")
except Exception as e:
    print(e)
    # warning: target log_prob returns NaN.
    # gets all-NaN tensor
# Inf tensor
try:
    # fill an x-like tensor with Inf
    target = lambda x: torch.tensor(torch.inf, dtype=torch.float32).to(device) * torch.ones(x.shape[0], dtype=torch.float32).to(device)
    results, _ = importance_sampling(100000, target, proposal, lambda x: x)
    print(f"Testing Inf tensor. Test mean:{results}")
except Exception as e:
    print(e)
    # gets all-NaN tensor
# -Inf tensor
try:
    # fill an x-like tensor with -Inf
    target = lambda x: torch.tensor(-torch.inf, dtype=torch.float32).to(device) * torch.ones(x.shape[0], dtype=torch.float32).to(device)
    results, _ = importance_sampling(100000, target, proposal, lambda x: x)
    print(f"Testing -Inf tensor. Test mean:{results}")
except Exception as e:
    print(e)
    # gets all-NaN tensor
# another -Inf test case, with mul_factor = 1
try:
    target = TensorizedMultiGauss(mean=[[[-torch.inf] * 2] * 2] * 3, std=torch.abs(0.8 * torch.ones(test_mean.shape)), device=device)
    target.mul_factor = 1.0
    results, _ = importance_sampling(100000, target, proposal, lambda x: x)
    print(f"Testing -Inf tensor(with mul_factor=1). Test mean:{results}")
except Exception as e:
    print(e)
    # gets zero tensor

print("")
# =============================================================================
# resampling
# =============================================================================

test_mean = [-1, 1, 0.5]

target = MultiGauss(mean=test_mean, std=[1, 1, 1])
# target.mul_factor = None

proposal = MultiGauss(mean=[0, 0, 0], std=[1, 1, 1])
expectation, test = importance_sampling(100000, target, proposal, eval_func=lambda x: abs(x), resample_ratio=0.5)
print(f"Evaluated mean:{expectation}")
print(f"Resampling mean:{torch.mean(test, dim=0)}, std:{torch.std(test, dim=0)}, count:{len(test)}")
print()
expectation, test = importance_sampling(100000, target, proposal, eval_func=lambda x: abs(x), resample_ratio=1)
print(f"Evaluated mean:{expectation}")
print(f"Resampling mean:{torch.mean(test, dim=0)}, std:{torch.std(test, dim=0)}, count:{len(test)}")
print()
expectation, test = importance_sampling(100000, target, proposal, eval_func=lambda x: abs(x), resample_ratio=0.00014)
print(f"Evaluated mean:{expectation}")
print(f"Resampling:{test}")
print()
expectation, test = importance_sampling(100000, target, proposal, eval_func=lambda x: abs(x), resample_ratio=1.9999999999999999/100000)
print(f"Evaluated mean:{expectation}")
print(f"Resampling:{test}")
print()
expectation, test = importance_sampling(100000, target, proposal, eval_func=lambda x: abs(x), resample_ratio=0.1/100000)
print(f"Evaluated mean:{expectation}")
print(f"Resampling:{test}")
print()
expectation, test = importance_sampling(100000, target, proposal, eval_func=lambda x: abs(x), resample_ratio=0)
print(f"Evaluated mean:{expectation}")
print(f"Resampling:{test}")
print()
try:
    expectation, test = importance_sampling(100000, target, proposal, eval_func=lambda x: abs(x), resample_ratio=-0.1)
    print(f"Evaluated mean:{expectation}")
    print(f"Resampling:{test}")
except Exception as e:
    print(e)
print()
try:
    expectation, test = importance_sampling(100000, target, proposal, eval_func=lambda x: abs(x), resample_ratio=1.1)
    print(f"Evaluated mean:{expectation}")
    print(f"Resampling:{test}")
except Exception as e:
    print(e)
print()
try:
    expectation, test = importance_sampling(100000, target, proposal, eval_func=lambda x: abs(x), resample_ratio=torch.inf)
    print(f"Evaluated mean:{expectation}")
    print(f"Resampling:{test}")
except Exception as e:
    print(e)
print()
try:
    expectation, test = importance_sampling(100000, target, proposal, eval_func=lambda x: abs(x), resample_ratio=-torch.inf)
    print(f"Evaluated mean:{expectation}")
    print(f"Resampling:{test}")
except Exception as e:
    print(e)
print()
try:
    expectation, test = importance_sampling(100000, target, proposal, eval_func=lambda x: abs(x), resample_ratio=torch.nan)
    print(f"Evaluated mean:{expectation}")
    print(f"Resampling:{test}")
except Exception as e:
    print(e)
