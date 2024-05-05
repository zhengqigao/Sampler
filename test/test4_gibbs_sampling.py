import torch
from sampler._common import Condistribution
from test_common_helper import ConditionalMultiGauss, UnconditionalMultiGauss
from sampler.base import *
import matplotlib.pyplot as plt

## TODO: use an example of joint Gaussian, so that we know all conditional distributions are Gaussian. Test Gibbs sampling on this example.
'''
gauss1 = ConditionalMultiGauss(std=[1])
gauss2 = ConditionalMultiGauss(std=[1])
results, info = gibbs_sampling(num_samples=10000, condis=[gauss1, gauss2], initial=torch.zeros(1,2))
print(f"Mean: {torch.mean(results, dim=0)}\tSize: {results.shape}")
plt.figure()
plt.scatter(results[:, 0], results[:, 1], s=1)
plt.show()
'''

gauss1 = ConditionalMultiGauss(std=[1, 1])
gauss2 = ConditionalMultiGauss(std=[1, 1])
gauss3 = ConditionalMultiGauss(std=[1, 1])
results, info = gibbs_sampling(num_samples=10000, condis=[gauss1, gauss2, gauss3], initial=torch.zeros(1,3))
print(f"Mean: {torch.mean(results, dim=0)}\tSize: {results.shape}")
plt.figure()
plt.scatter(results[:, 0], results[:, 1], s=1)
plt.show()
