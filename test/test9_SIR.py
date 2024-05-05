import torch

from sampler.base import *
from sampler._common import Distribution
from sampler.distribution import TDWrapper
from test_common_helper import MultiGauss, TensorizedMultiGauss

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
