from sampler.base import *
from sampler._common import Distribution
from test_common_helper import MultiGauss, DensityFunc

# try:
#     target = MultiGauss(mean=test_mean, std=[1, 1, 1])
#     target.mul_factor = None
#     proposal = MultiGauss(mean=[0, 0, 0], std=[1, 1, 1])
#     results, info = rejection_sampling(10000, target, proposal, k=1000)
#     print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
# except Exception as e:
#     print(e)
# # Not always successful.
# # Because both target and proposal are Gaussian,
# # thus proposal std should be greater than target std,
# # otherwise k*proposal could never cover target.

test_mean = [-1, 1, 0.5]
target = MultiGauss(mean=test_mean, std=[1, 1, 1])
proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
results, info = rejection_sampling(10000, target, proposal, k=40)
print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")

proposal = MultiGauss(mean=[0, 0, 0], std=[2, 2, 2])
results, info = rejection_sampling(10000, target, proposal, k=155)
print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")

proposal = MultiGauss(mean=[0, 0, 0], std=[3, 3, 3])
results, info = rejection_sampling(10000, target, proposal, k=900)
print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")


print("=====================================")

# change mul_factor of target

test_mean = [-1, 1, 0.5]
target = MultiGauss(mean=test_mean, std=[1, 1, 1])
proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
try:
    target.mul_factor = None
    results, info = rejection_sampling(10000, target, proposal, k=40)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # nothing changed
try:
    target.mul_factor = 0
    results, info = rejection_sampling(10000, target, proposal, k=40)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # raised ValueError: The mul_factor must be a positive limited scalar, but got 0.
try:
    target.mul_factor = 33.3
    results, info = rejection_sampling(10000, target, proposal, k=40)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # raised ValueError: The scaling factor k = 40 is not large enough.
try:
    target.mul_factor = 1 / 33.3
    results, info = rejection_sampling(10000, target, proposal, k=40)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # runs slower, rejection rate rises from 0.975 to 0.999

print("=====================================")

# weird target objects
# e.g. invalid target obj, target obj without p.d.f., target obj without mul_factor
try:
    target = MultivariateNormal(torch.tensor([0, 0], dtype=torch.float32), torch.eye(2))
    proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
    results, info = rejection_sampling(10000, target, proposal, k=100)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # raised: 'MultivariateNormal' object is not callable
try:
    target = Distribution()
    proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
    results, info = rejection_sampling(10000, target, proposal, k=100)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # raised NotImplementedError (printed an empty line)
try:
    target = MultiGauss(mean=[0, 0], std=[1, 1])
    target.log_prob = None
    proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
    results, info = rejection_sampling(10000, target, proposal, k=100)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # raised: 'NoneType' object is not callable
try:
    target = None
    proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
    results, info = rejection_sampling(10000, target, proposal, k=100)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # raised: 'NoneType' object is not callable
try:
    target = MultiGauss(mean=[0, 0, 0], std=[1, 1, 1])
    proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
    results, info = rejection_sampling(10000, target, proposal, k=100)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # raised: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
try:
    target = MultiGauss(mean=[0, 0], std=[1, 1])
    proposal = MultiGauss(mean=[0, 0, 0], std=[1.5, 1.5, 1.5])
    results, info = rejection_sampling(10000, target, proposal, k=100)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # raised: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1

print("=====================================")

# target functions
try:
    test_mean = [-1, 1, 0.5]
    target = lambda x: -0.5 * (torch.sum(((x - torch.tensor(test_mean, dtype=torch.float32)) / torch.tensor([1, 1, 1], dtype=torch.float32)) ** 2, dim=1) + torch.log(2 * torch.pi * torch.tensor([1, 1, 1], dtype=torch.float32) ** 2).sum())
    proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
    results, info = rejection_sampling(10000, target, proposal, k=40)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)

# custom target functions
try:
    proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
    results, info = rejection_sampling(10000, DensityFunc("potential1"), proposal, k=200)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
try:
    proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
    results, info = rejection_sampling(10000, DensityFunc("potential6"), proposal, k=200)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
try:
    proposal = MultiGauss(mean=[0, 0], std=[2, 2])
    results, info = rejection_sampling(10000, DensityFunc("potential7"), proposal, k=200)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)

# weird target functions
try:  # target function returning None
    target = lambda x: torch.tensor(None, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
    results, info = rejection_sampling(10000, target, proposal, k=100)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # raised TypeError: must be real number, not NoneType
try:  # target function returning NaN
    target = lambda x: torch.tensor(torch.nan, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
    results, info = rejection_sampling(10000, target, proposal, k=100)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # [nan, nan]
try:  # target function returning Inf
    target = lambda x: torch.tensor(torch.inf, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
    results, info = rejection_sampling(10000, target, proposal, k=100)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # The scaling factor k = 100 is not large enough.
    # in fact, the target p.d.f. is always Inf, thus no k would be large enough
try:
    print("This test case will run into dead loop, please stop it manually.")
    # target function returning -Inf, meaning target p.d.f. is always 0
    # resulting in dead loop, because RS continues to sample
    # until k samples are accepted, while no sample can be accepted
    # THIS IS FEATURE, NOT BUG
    target = lambda x: torch.tensor(-torch.inf, dtype=torch.float32) * torch.ones(x.shape[0], dtype=torch.float32)
    proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
    results, info = rejection_sampling(10000, target, proposal, k=100, max_samples=100000)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except KeyboardInterrupt:
    print("KeyboardInterrupt")
except Exception as e:
    print(e)
    # UserWarning: Rejection sampling reaches the maximum number of samples: 100000.
    #   warnings.warn(f"Rejection sampling reaches the maximum number of samples: {max_samples}.")
    # Mean: tensor([nan, nan])        Rejection rate: 1.0     Size: torch.Size([0, 2])
try:
    target = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
    proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
    results, info = rejection_sampling(10000, target, proposal, k=100, max_samples=10)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except KeyboardInterrupt:
    print("KeyboardInterrupt")
except Exception as e:
    print(e)
    # UserWarning: Rejection sampling reaches the maximum number of samples: 10.
    #   warnings.warn(f"Rejection sampling reaches the maximum number of samples: {max_samples}.")
    # Mean: tensor([ 0.1072, -0.2177])        Rejection rate: 0.9899  Size: torch.Size([101, 2])

print("=====================================")

# weird cases for k
try:
    test_mean = [-1, 1, 0.5]
    target = MultiGauss(mean=test_mean, std=[1, 1, 1])
    proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
    results, info = rejection_sampling(10000, target, proposal, k=0)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # raised ValueError: The scaling factor k should be a positive finite scalar, but got k = 0.
try:
    test_mean = [-1, 1, 0.5]
    target = MultiGauss(mean=test_mean, std=[1, 1, 1])
    proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
    results, info = rejection_sampling(10000, target, proposal, k=-100)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # raised ValueError: The scaling factor k should be a positive finite scalar, but got k = -100.
try:
    test_mean = [-1, 1, 0.5]
    target = MultiGauss(mean=test_mean, std=[1, 1, 1])
    proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
    results, info = rejection_sampling(10000, target, proposal, k=-torch.inf)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # raised ValueError: The scaling factor k should be a positive finite scalar, but got k = -inf.
try:
    test_mean = [-1, 1, 0.5]
    target = MultiGauss(mean=test_mean, std=[1, 1, 1])
    proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
    results, info = rejection_sampling(10000, target, proposal, k=40.5)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # float k is accepted
try:
    test_mean = [-1, 1, 0.5]
    target = MultiGauss(mean=test_mean, std=[1, 1, 1])
    proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
    results, info = rejection_sampling(10000, target, proposal, k=torch.inf)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # raised ValueError: The scaling factor k should be a positive finite scalar, but got k = inf.
try:
    test_mean = [-1, 1, 0.5]
    target = MultiGauss(mean=test_mean, std=[1, 1, 1])
    proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
    results, info = rejection_sampling(10000, target, proposal, k=torch.nan)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    # raised ValueError: The scaling factor k should be a positive finite scalar, but got k = nan.
