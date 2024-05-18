import numpy as np
import matplotlib.pyplot as plt
from sampler.base import *
from sampler._common import Distribution
from test_common_helper import MultiGauss, DensityFunc

def show(target, results, info):
    x = torch.linspace(-5, 5, 100)
    y = torch.linspace(-5, 5, 100)
    xx, yy = torch.meshgrid(x, y)
    grid_data = torch.cat((yy.reshape(-1, 1), xx.reshape(-1, 1)), dim=1)
    value = target(grid_data)
    z = torch.exp(value).reshape(100,-1)
    plt.figure()
    plt.title(f"Mean: {torch.mean(results, dim=0)}\nRejection rate: {info['rejection_rate']}")
    plt.pcolormesh(x, y, z[:-1, :-1], cmap='summer')
    plt.colorbar()
    plt.scatter(results[:, 0], results[:, 1], marker='.', color='red', alpha=0.1)
    plt.show()

test_mean = [-1, 0.5]
target = MultiGauss(mean=test_mean, std=[1, 1])
proposal = MultiGauss(mean=[0, 0], std=[1.2, 1.2])
squeezing = MultiGauss(mean=[1, -1], std=[0.9, 0.9])
results, info = rejection_sampling(2000, target,
                                   proposal=proposal, k=6,
                                   squeezing=squeezing, k_squeezing=0.03)
print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
show(target, results, info)

test_mean = [-1, 0.5]
target = MultiGauss(mean=test_mean, std=[1, 1])
proposal = MultiGauss(mean=[0, 0], std=[1.2, 1.2])
squeezing = MultiGauss(mean=[1, -1], std=[0.9, 0.9])
results, info = rejection_sampling(2000, target,
                                   proposal=proposal, k=6,
                                   squeezing=squeezing, k_squeezing=0.5)
print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
show(target, results, info)
