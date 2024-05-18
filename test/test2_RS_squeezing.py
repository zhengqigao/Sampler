from sampler.base import *
from sampler._common import Distribution
from test_common_helper import MultiGauss, DensityFunc

test_mean = [-1, 1, 0.5]
target = MultiGauss(mean=test_mean, std=[1, 1, 1])
proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
squeezing = MultiGauss(mean=test_mean, std=[0.6, 0.6, 0.6])
results, info = rejection_sampling(10000, target,
                                   proposal=proposal, k=40,
                                   squeezing=squeezing, k_squeezing=0.7)
print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
