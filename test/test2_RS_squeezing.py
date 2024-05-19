import numpy as np
import matplotlib.pyplot as plt
from sampler.base import *
from sampler._common import Distribution
from test_common_helper import MultiGauss, DensityFunc, PlotSamples

test_mean = [-1, 0.5]
target = MultiGauss(mean=test_mean, std=[1, 1])
proposal = MultiGauss(mean=[0, 0], std=[1.2, 1.2])
squeezing = MultiGauss(mean=[-1, 1], std=[0.9, 0.9])
results, info = rejection_sampling(2000, target,
                                   proposal=proposal, k=6,
                                   squeezing=squeezing, k_squeezing=0.1)
print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
PlotSamples(target, results, info)

try:
    test_mean = [-1, 0.5]
    target = MultiGauss(mean=test_mean, std=[1, 1])
    proposal = MultiGauss(mean=[0, 0], std=[1.2, 1.2])
    squeezing = MultiGauss(mean=[1, -1], std=[0.9, 0.9])
    results, info = rejection_sampling(2000, target,
                                    proposal=proposal, k=6,
                                    squeezing=squeezing, k_squeezing=0.5)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
    PlotSamples(target, results, info)
except Exception as e:
    print(e)

target = DensityFunc("potential6")
proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
results, info = rejection_sampling(2000, target, proposal, k=200)
print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
PlotSamples(target, results, info)

target = DensityFunc("potential6")
proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
squeezing = MultiGauss(mean=[-1.5, 0], std=[1, 1])
results, info = rejection_sampling(2000, target, proposal, k=200, squeezing=squeezing, k_squeezing=0.4)
print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
PlotSamples(target, results, info)
# RuntimeError: cannot reshape tensor of 0 elements into shape [0, -1] because the unspecified dimension size -1 can be any value and is ambiguous
# ??? occasionally raise error

target = DensityFunc("potential1")
proposal = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
results, info = rejection_sampling(2000, target, proposal, k=200)
print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
PlotSamples(target, results, info)