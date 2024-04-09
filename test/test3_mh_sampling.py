from sampler.base import *
import torch
from sampler._common import Distribution, Condistribution
import matplotlib.pyplot as plt


class ConditionalMultiGauss(Condistribution):
    def __init__(self, std):
        super().__init__()
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(std)
        self.mul_factor = 1.0

    def sample(self, num_samples: int, y) -> torch.Tensor:
        # y has shape (m, d)
        # return shape (num_samples, m, d) with y as the mean
        assert len(y.shape) == 2 and y.shape[1] == self.dim
        return torch.randn((num_samples, y.shape[0], y.shape[1])) * self.std + y

    def evaluate_density(self, x: torch.Tensor, y: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        # x is of shape (N,d), y is of shape (M,d)
        # return shape (N,M)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        if in_log:
            return -0.5 * (torch.sum(((x - y) / self.std) ** 2, dim=2) + torch.log(2 * torch.pi * self.std * self.std).sum())
        else:
            return torch.exp(-0.5 * torch.sum(((x - y) / self.std) ** 2, dim=2)) / (
                    torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim * torch.prod(self.std * self.std))

class UnconditionalMultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(std)
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        if in_log:
            return -0.5 * (torch.sum(((x - self.mean) / self.std) ** 2, dim=1) + torch.log(2 * torch.pi * self.std * self.std).sum())
        else:
            return torch.exp(-0.5 * torch.sum(((x - self.mean) / self.std) ** 2, dim=1)) / (
                    torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim * torch.prod(self.std * self.std))

#This case is extended with event_func test case
gauss1 = ConditionalMultiGauss(std = [1, 1])
gauss2 = UnconditionalMultiGauss(mean=[-2,2], std=[1, 1])
results, info = mh_sampling(50000, gauss2, gauss1, torch.zeros(3,2), burn_in=10000,event_func=lambda sample:sample[-1][0][0]>2) # 3 different MC chains, each grown by MH independently

for batch_index in range(results.shape[1]):
    plt.figure()
    plt.scatter(results[:, batch_index, 0], results[:, batch_index, 1], s=1)
plt.show()

print(f"info['acceptance_rate'] = {info['acceptance_rate']}")


# TODO: zhengqi: I tested potential4, potential3 and potential6. Can you test the others? Also, I think potential6
#  looks a bit weird. Can you check it? 3 and 4 look fine to me.
# Nanlin: I have tested Potential 1-7 and I think all cases are Great. BTW, how can you type yellow words,
# I wanna learn~

from test_common_helper import PotentialFunc
potential_name = "potential3"
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
tmp, _ = mh_sampling(50000, target=lambda x, in_log: -potential_func(x, True),
                     transit=ConditionalMultiGauss(torch.ones(2)), initial=torch.zeros((1, 2)), burn_in=5000)
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