from sampler.base import *
import torch
from sampler._common import Distribution, Condistribution
import matplotlib.pyplot as plt
import time

from test_common_helper import ConditionalMultiGauss,UnconditionalMultiGauss
#This case is extended with event_func test case
gauss1 = ConditionalMultiGauss(std = [1, 1])
gauss2 = UnconditionalMultiGauss(mean=[-2,2], std=[1, 1])
results, info = mh_sampling(50000, gauss2, gauss1, torch.zeros(3,2), burn_in=10000,event_func=lambda sample:sample[-1][0][0]>2) # 3 different MC chains, each grown by MH independently

for batch_index in range(results.shape[1]):
    plt.figure()
    plt.scatter(results[:, batch_index, 0], results[:, batch_index, 1], s=1)
plt.show()

print(f"info['acceptance_rate'] = {info['acceptance_rate']}")



from test_common_helper import PotentialFunc
#TODO: Nanlin: Some numerical error happens for potential 3,4, I'll check it later, the num of sample fall in the region is small
potential_name = "potential4"
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

# sample by mh_sampling
tmp, _ = mh_sampling(50000, target=lambda x: -potential_func(x),
                     transit=ConditionalMultiGauss(torch.ones(2)), initial=torch.ones((1, 2)), burn_in=5000)

print(tmp.shape)
plt.figure()
plt.title("test result "+potential_name)
# only show samples within bound
tmp = tmp[tmp[:, 0, 0] > -bound]
tmp = tmp[tmp[:, 0, 0] < bound]
tmp = tmp[tmp[:, 0, 1] > -bound]
tmp = tmp[tmp[:, 0, 1] < bound]
print(tmp.shape)
plt.scatter(tmp[:, 0, 0], tmp[:, 0, 1], s=1)
plt.xlim(-bound, bound)
plt.ylim(-bound, bound)
plt.show()


###test for GPU version
from test_common_helper import TensorizedMultiGauss,TensorizedConditionalMultiGauss

test_mean = torch.tensor([-2,2])
test_std = torch.tensor([1,1])
# Please test the algorithm on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps") Nanlin: this is for my arm GPU

start_time = time.time()
gauss2 = TensorizedMultiGauss(mean=test_mean, std=test_std, device=device)
gauss1 = TensorizedConditionalMultiGauss(std = [1, 1],device=device)
results, info = mh_sampling(10000, gauss2, gauss1, torch.zeros(1,2).to(device), burn_in=0) # 3 different MC chains, each grown by MH independently
results = results.cpu()
for batch_index in range(results.shape[1]):
    plt.figure()
    plt.scatter(results[:, batch_index, 0], results[:, batch_index, 1], s=1)
plt.show()
end_time = time.time()
time_interval = end_time-start_time
print(f"running time interval for tensor version is {time_interval}")

