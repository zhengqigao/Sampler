import torch

from sampler.base import *
from sampler._common import Distribution
from test_common_helper import MultiGauss
import matplotlib.pyplot as plt
import seaborn as sns


test_mean = [-1]
test_std = [0.5]
lower = -1.3
upper = -0.2
target = MultiGauss(mean=test_mean, std=test_std)
target.mul_factor = None

results, info = adaptive_rejection_sampling(10000, target, lower, upper)

print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\t#Iteration: {info['iteration_count']}\tSize: {results.shape}")

plt.subplot(1,2,1)
plt.hist(results.cpu().detach().numpy(), bins=50)
plt.subplot(1,2,2)
sns.kdeplot(data=results.cpu().detach().numpy())
plt.show()


print("=====================================")

test_mean = [1]
test_std = [1]
lower = 0.5
upper = 1.5
target = MultiGauss(mean=test_mean, std=test_std)
target.mul_factor = None

results, info = adaptive_rejection_sampling(10000, target, lower, upper)

print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\t#Iteration: {info['iteration_count']}\tSize: {results.shape}")

plt.subplot(1,2,1)
plt.hist(results.cpu().detach().numpy(), bins=50)
plt.subplot(1,2,2)
sns.kdeplot(data=results.cpu().detach().numpy())
plt.show()
