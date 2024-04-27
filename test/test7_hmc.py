from sampler.base import *
import torch
from sampler._common import Distribution, Condistribution
import matplotlib.pyplot as plt
from test_common_helper import UnconditionalMultiGauss

# Simple case

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


# Cuda function test
from test_common_helper import TensorizedMultiGauss,TensorizedConditionalMultiGauss

test_mean = torch.tensor([-2,2])
test_std = torch.tensor([1,1])
# Please test the algorithm on GPU if available
# TODO: Nanlin: Can anyone help me test this case? It is not supported for the ARM device yet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")  #Nanlin: this is for my arm GPU

gauss2 = TensorizedMultiGauss(mean=test_mean, std=test_std, device=device)
samples = hamiltonian_monte_carlo(num_samples=10000,
                               target=gauss2,
                               step_size=0.1,
                               num_leapfrog=10,
                               initial=torch.zeros(1,2).to(device),
                               burn_in=0)

samples = samples.cpu()
for i in range(samples.shape[1]):
    plt.figure()
    plt.scatter(samples[:,i, 0], samples[:, i, 1], s=1)
plt.show()

# Potential function test

from test_common_helper import PotentialFunc
potential_name = "potential6"
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

tmp = hamiltonian_monte_carlo(num_samples=10000,
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