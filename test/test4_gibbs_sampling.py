import torch
from sampler._common import Condistribution
from test_common_helper import MultiGauss, CorMultiGauss2D, CondGaussGauss1D, BlockCondGaussGauss
from sampler.base import *
import matplotlib.pyplot as plt
from test_common_helper import CondGaussGamma, CondGammaGauss
import seaborn as sns

## TODO: use an example of joint Gaussian, so that we know all conditional distributions are Gaussian. Test Gibbs sampling on this example.


print("=====================================")
print("Test 1: Normal Likelihood")
print("=====================================")
mean = [4]
std = [2]
target = MultiGauss(mean=mean, std=std)
print(f"Expected mean: {mean}, expected std: {std}")
data = target.sample(1000)
w = 1
alpha = 2
beta = 1
gauss1 = CondGaussGamma(data, w)
gauss2 = CondGammaGauss(data, alpha, beta)
results, info = gibbs_sampling(num_samples=10000, condis=[gauss1, gauss2], initial=torch.zeros(1,2), burn_in=1000)
results[:, 1] = 1/results[:, 1].sqrt()
print(f"Mean: {torch.mean(results, dim=0)}\tSize: {results.shape}")
# The results should be the prediction of mean and 1/std
plt.figure()
plt.scatter(results[:, 0], results[:, 1], s=1)
plt.show()

# joint distribution based on Gibbs sampling
g = sns.jointplot(x=results[:,0], y=results[:,1], alpha=0.5, kind='hex');
g.plot_joint(sns.kdeplot, color="r", zorder=1, levels=4);
g.set_axis_labels('$\mu$', '$\sigma$', fontsize=16);
plt.show()

print("=====================================")
print("Test 2: Bivariate Normal Distribution")
print("=====================================")

mean = [2, 1]
std = [1, 4]
rho = 0
target = CorMultiGauss2D(mean=mean, std=std, rho=rho)
print(f"Expected mean: {mean}, expected std: {std}, expected rho: {rho}")
gauss1 = CondGaussGauss1D(mean=[2], std=[1], mean_cond=[1], std_cond=[4], rho=0)
gauss2 = CondGaussGauss1D(mean=[1], std=[4], mean_cond=[2], std_cond=[1], rho=0)
results, info = gibbs_sampling(num_samples=10000, condis=[gauss1, gauss2], initial=torch.zeros(1,2))
print(f"Mean: {torch.mean(results, dim=0)}\tSize: {results.shape}")
plt.figure()
plt.xlim(-14,16)
plt.ylim(-13,17)
plt.scatter(results[:, 0], results[:, 1], s=1)
plt.show()

# joint distribution based on Gibbs sampling
g = sns.jointplot(x=results[:,0], y=results[:,1], xlim=[-14,16], ylim=[-13,17], alpha=0.5, kind='hex');
g.plot_joint(sns.kdeplot, color="r", zorder=1, levels=4);
g.set_axis_labels('$\mu_1$', '$\mu_2$', fontsize=16);
plt.show()

print("=====================================")
print("Test 3: Bivariate Normal Distribution")
print("=====================================")

mean = [2, 1]
std = [3, 2]
rho = 0.9
target = CorMultiGauss2D(mean=[2,1], std=[3,2], rho=0.9)
print(f"Expected mean: {mean}, expected std: {std}, expected rho: {rho}")
gauss1 = CondGaussGauss1D(mean=[2], std=[3], mean_cond=[1], std_cond=[2], rho=0.9)
gauss2 = CondGaussGauss1D(mean=[1], std=[2], mean_cond=[2], std_cond=[3], rho=0.9)
results, info = gibbs_sampling(num_samples=10000, condis=[gauss1, gauss2], initial=torch.zeros(1,2))
print(f"Mean: {torch.mean(results, dim=0)}\tSize: {results.shape}")
plt.figure()
plt.xlim(-14,16)
plt.ylim(-13,17)
plt.scatter(results[:, 0], results[:, 1], s=1)
plt.show()

# joint distribution based on Gibbs sampling
g = sns.jointplot(x=results[:,0], y=results[:,1], xlim=[-14,16], ylim=[-13,17], alpha=0.5, kind='hex');
g.plot_joint(sns.kdeplot, color="r", zorder=1, levels=4);
g.set_axis_labels('$\mu_1$', '$\mu_2$', fontsize=16);
plt.show()

print("=====================================")
print("Test 4: Bivariate Normal Distribution (Blocking)")
print("=====================================")

mean = [2, 1]
std = [3, 2]
rho = 0.9
target = CorMultiGauss2D(mean=mean, std=std, rho=rho)
print(f"Expected mean: {mean}, expected std: {std}, expected rho: {rho}")
gauss1 = CondGaussGauss1D(mean=[2], std=[3], mean_cond=[1], std_cond=[2], rho=0.9)
gauss2 = CondGaussGauss1D(mean=[1], std=[2], mean_cond=[2], std_cond=[3], rho=0.9)
results, info = gibbs_sampling(num_samples=10000, condis=[gauss1, gauss2], initial=torch.zeros(1,2), block=True, block_list=[[0,0],[1,1]])
print(f"Mean: {torch.mean(results, dim=0)}\tSize: {results.shape}")
plt.figure()
plt.xlim(-14,16)
plt.ylim(-13,17)
plt.scatter(results[:, 0], results[:, 1], s=1)
plt.show()

# joint distribution based on Gibbs sampling
g = sns.jointplot(x=results[:,0], y=results[:,1], xlim=[-14,16], ylim=[-13,17], alpha=0.5, kind='hex');
g.plot_joint(sns.kdeplot, color="r", zorder=1, levels=4);
g.set_axis_labels('$\mu_1$', '$\mu_2$', fontsize=16);
plt.show()

print("=====================================")
print("Test 5: Normal Distribution (Blocking)")
print("=====================================")

mean_target = [2, 2, 2, 2]
sigma_target = [[1, 0, 0.5, 0], [0, 1, 0, 0], [0.5, 0, 1, 0], [0, 0, 0, 1]]
print(f"Expected mean: {mean_target}")
# target distribution is a Multivariate Gaussian Distribution with aforementioned mean and sigma
gauss1 = BlockCondGaussGauss(mean_a=mean_target[0:2], mean_b=mean_target[2:4], sigma_aa=[[1,0],[0,1]], sigma_bb=[[1,0],[0,1]], sigma_ab=[[0.5,0],[0,0]], sigma_ba=[[0.5,0],[0,0]])
gauss2 = BlockCondGaussGauss(mean_a=mean_target[2:4], mean_b=mean_target[0:2], sigma_aa=[[1,0],[0,1]], sigma_bb=[[1,0],[0,1]], sigma_ab=[[0.5,0],[0,0]], sigma_ba=[[0.5,0],[0,0]])
results, info = gibbs_sampling(num_samples=10000, condis=[gauss1, gauss2], initial=torch.zeros(1, 4), block=True,
                               block_list=[[0, 1], [2, 3]])
print(f"Mean: {torch.mean(results, dim=0)}\tSize: {results.shape}")
plt.figure()
plt.xlim(-3, 7)
plt.ylim(-3, 7)
plt.scatter(results[:, 0], results[:, 1], s=1)
plt.show()

# joint distribution based on Gibbs sampling
print(f"Expected mean: {mean_target[0]}, {mean_target[1]}")
g = sns.jointplot(x=results[:, 0], y=results[:, 1], xlim=[-3, 7], ylim=[-3, 7], alpha=0.5, kind='hex')
g.plot_joint(sns.kdeplot, color="r", zorder=1, levels=4)
g.set_axis_labels('$\mu_1$', '$\mu_2$', fontsize=16)
plt.show()

print(f"Expected mean: {mean_target[0]}, {mean_target[2]}")
g = sns.jointplot(x=results[:, 0], y=results[:, 2], xlim=[-3, 7], ylim=[-3, 7], alpha=0.5, kind='hex')
g.plot_joint(sns.kdeplot, color="r", zorder=1, levels=4)
g.set_axis_labels('$\mu_1$', '$\mu_3$', fontsize=16)
plt.show()

print("=====================================")
print("Test 6: Normal Distribution (Blocking)")
print("=====================================")

mean_target = [2, 2, 2, 2]
sigma_target = [[3, 0, 2.7, 0], [0, 3, 0, 0], [2.7, 0, 3, 0], [0, 0, 0, 3]]
print(f"Expected mean: {mean_target}")
# target distribution is a Multivariate Gaussian Distribution with aforementioned mean and sigma
gauss1 = BlockCondGaussGauss(mean_a=mean_target[0:2], mean_b=mean_target[2:4], sigma_aa=[[3,0],[0,3]], sigma_bb=[[3,0],[0,3]], sigma_ab=[[2.7,0],[0,0]], sigma_ba=[[2.7,0],[0,0]])
gauss2 = BlockCondGaussGauss(mean_a=mean_target[2:4], mean_b=mean_target[0:2], sigma_aa=[[3,0],[0,3]], sigma_bb=[[3,0],[0,3]], sigma_ab=[[2.7,0],[0,0]], sigma_ba=[[2.7,0],[0,0]])
results, info = gibbs_sampling(num_samples=10000, condis=[gauss1, gauss2], initial=torch.zeros(1, 4), block=True,
                               block_list=[[0, 1], [2, 3]])
print(f"Mean: {torch.mean(results, dim=0)}\tSize: {results.shape}")
plt.figure()
plt.xlim(-14, 16)
plt.ylim(-13, 17)
plt.scatter(results[:, 0], results[:, 1], s=1)
plt.show()

# joint distribution based on Gibbs sampling
print(f"Expected mean: {mean_target[0]}, {mean_target[1]}")
g = sns.jointplot(x=results[:, 0], y=results[:, 1], xlim=[-14, 16], ylim=[-13, 17], alpha=0.5, kind='hex')
g.plot_joint(sns.kdeplot, color="r", zorder=1, levels=4)
g.set_axis_labels('$\mu_1$', '$\mu_2$', fontsize=16)
plt.show()

print(f"Expected mean: {mean_target[0]}, {mean_target[2]}")
g = sns.jointplot(x=results[:, 0], y=results[:, 2], xlim=[-14, 16], ylim=[-13, 17], alpha=0.5, kind='hex')
g.plot_joint(sns.kdeplot, color="r", zorder=1, levels=4)
g.set_axis_labels('$\mu_1$', '$\mu_3$', fontsize=16)
plt.show()
