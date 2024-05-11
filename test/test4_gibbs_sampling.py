import torch
from sampler._common import Condistribution
from test_common_helper import ConditionalMultiGauss, UnconditionalMultiGauss, MultiGauss, CorMultiGauss, CondGaussGauss
from sampler.base import *
import matplotlib.pyplot as plt
from test_common_helper import CondGaussGamma, CondGammaGauss


## TODO: use an example of joint Gaussian, so that we know all conditional distributions are Gaussian. Test Gibbs sampling on this example.

'''
target = MultiGauss(mean=[2], std=[1])
data = target.sample(100)
w = 1
alpha = 2
beta = 1
gauss1 = CondGaussGamma(data, w)
gauss2 = CondGammaGauss(data, alpha, beta)
results, info = gibbs_sampling(num_samples=10000, condis=[gauss1, gauss2], initial=torch.zeros(1,2), burn_in=1000)
print(f"Mean: {torch.mean(results, dim=0)}\tSize: {results.shape}")
# The results should be the prediction of mean and 1/std
plt.figure()
plt.scatter(results[:, 0], results[:, 1], s=1)
plt.show()
'''

target = CorMultiGauss(mean=[2,1], std=[1,4], rho=0)
gauss1 = CondGaussGauss(mean=[2], std=[1], mean_cond=[1], std_cond=[4], rho=0)
gauss2 = CondGaussGauss(mean=[1], std=[4], mean_cond=[2], std_cond=[1], rho=0)
results, info = gibbs_sampling(num_samples=10000, condis=[gauss1, gauss2], initial=torch.zeros(1,2))
print(f"Mean: {torch.mean(results, dim=0)}\tSize: {results.shape}")
plt.figure()
plt.xlim(-8,12)
plt.ylim(-13,17)
plt.scatter(results[:, 0], results[:, 1], s=1)
plt.show()


target = CorMultiGauss(mean=[2,1], std=[3,2], rho=0.9)
gauss1 = CondGaussGauss(mean=[2], std=[3], mean_cond=[1], std_cond=[2], rho=0.9)
gauss2 = CondGaussGauss(mean=[1], std=[2], mean_cond=[2], std_cond=[3], rho=0.9)
results, info = gibbs_sampling(num_samples=10000, condis=[gauss1, gauss2], initial=torch.zeros(1,2))
print(f"Mean: {torch.mean(results, dim=0)}\tSize: {results.shape}")
plt.figure()
plt.xlim(-8,12)
plt.ylim(-13,17)
plt.scatter(results[:, 0], results[:, 1], s=1)
plt.show()
