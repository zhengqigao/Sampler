import torch

from sampler.base import *
from sampler._common import Distribution
from test_common_helper import MultiGauss


test_mean = [1]
test_std = [1]
lower = -1
upper = 3
target = MultiGauss(mean=test_mean, std=test_std)
target.mul_factor = None

results, info = adaptive_rejection_sampling(10000, target, lower, upper)

print(results.shape)
print(torch.mean(results, dim=0))
print(info['rejection_rate'])


