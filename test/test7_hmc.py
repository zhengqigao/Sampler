from sampler.base import *
import torch
from sampler._common import Distribution, Condistribution
import matplotlib.pyplot as plt
from test_common_helper import UnconditionalMultiGauss

### Simple case
samples = hamiltonian_monte_carlo(num_samples=10000,
                               target=UnconditionalMultiGauss([2, -2], [1, 1]),
                               step_size=0.1,
                               num_leapfrog=10,
                               initial=torch.rand(1, 2), # three different MC chains, sample independently
                               burn_in=0)


for i in range(samples.shape[1]):
    plt.figure()
    plt.scatter(samples[:,i, 0], samples[:, i, 1], s=1)
plt.show()

print(samples.shape)


### Potential function test

from test_common_helper import PotentialFunc
potential_name = "potential7"
potential_func = PotentialFunc(potential_name)
bound = 4
x = torch.linspace(-bound, bound, 100)
y = torch.linspace(-bound, bound, 100)
xx, yy = torch.meshgrid(x, y)
grid_data = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)

value = potential_func(grid_data)

# scatter them to see the potential on a heatmap
# For precise description, say we denote: distribution = exp(density) = exp(-potential)
# In the evaluate_density function, we need to provide `distribution` when in_log=False, and `density` when in_log=True®
# However, in the potential_func defined in test_common_helper, it actually returns the `potential` not the `density`.
plt.figure()
plt.scatter(grid_data[:, 0], grid_data[:, 1], c=torch.exp(-value), cmap='viridis')
plt.title('golden result '+potential_name)

# sample by LMC

#tmp, _ = mh_sampling(50000, target=lambda x: -potential_func(x),
#                     transit=ConditionalMultiGauss(torch.ones(2)), initial=torch.zeros((1, 2)), burn_in=5000)
tmp=hamiltonian_monte_carlo(num_samples=10000,
                               target=lambda x: -potential_func(x),
                               step_size=0.1,
                               num_leapfrog=10,
                               initial=torch.rand(1, 2),  # three different MC chains, sample independently
                               burn_in=0)

plt.figure()
plt.title("test result "+potential_name)
# only show samples within bound
tmp = tmp[tmp[:, 0, 0] > -bound]
tmp = tmp[tmp[:, 0, 0] < bound]
tmp = tmp[tmp[:, 0, 1] > -bound]
tmp = tmp[tmp[:, 0, 1] < bound]
plt.scatter(tmp[:, 0, 0], tmp[:, 0, 1], s=1)
plt.xlim(-bound, bound)
plt.ylim(-bound, bound)
plt.show()