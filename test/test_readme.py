# import torch
# from torch.distributions import MultivariateNormal
# from sampler.base import importance_sampling
# from sampler import Distribution, checker
# from sampler.distribution import TDWrapper
#
# # define the target and proposal distributions using the Wrapper
# target = TDWrapper(MultivariateNormal(torch.Tensor([-1, 1, 0.5]), torch.eye(3)))
# proposal = TDWrapper(MultivariateNormal(torch.zeros(3), torch.eye(3)))
#
# # define a function to estimate the expectation of. It can be a multidimensional function.
# # In this example, we consider a R^3 to R^3 identity function.
# f = lambda x: x
#
# # use proposal to estimate the expectation of f with respect to the target
# # As in this case f(x)=x, we essentially estimate the mean of the target distribution.
# results, _ = importance_sampling(10000, target, proposal, f)
# print("Test mean:", results)
#
# # Or we can directly define a distribution class in Sampler library.
# # The 'sample' and 'log_prob' methods need to be implemented by users.
# # It is suggested to add the checker decorator to enable automatic checking.
# class MultiGauss(Distribution):
#     def __init__(self, mean, std):
#         super().__init__()
#         self.mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)
#         self.std = std if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)
#         self.dim = len(self.mean)
#         self.mul_factor = 1.0
#
#     @checker
#     def sample(self, num_samples: int) -> torch.Tensor:
#         return torch.randn((num_samples, self.dim)) * self.std + self.mean
#
#     @checker
#     def log_prob(self, x: torch.Tensor) -> torch.Tensor:
#         return -0.5 * (
#                 torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
#                 + torch.log(2 * torch.pi * self.std * self.std).sum()
#         )
# target = MultiGauss([-1, 1, 0.5], [1,1,1])
# proposal = MultiGauss([0, 0, 0], [1,1,1])
# results, _ = importance_sampling(10000, target, proposal, f)
# print("Test mean:", results)

#
# import torch
# from sampler import Condistribution, checker
# from sampler.base import mh_sampling
# from math import pi
#
#
# # define a conditional distribution class in Sampler library.
# # The 'sample' and 'log_prob' methods need to be implemented by users.
# # It is suggested to add the checker decorator to enable automatic checking.
# class ConditionalMultiGauss(Condistribution):
#     def __init__(self, std):
#         super().__init__()
#         self.std = std if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)
#         self.dim = len(std)
#
#     @checker
#     def sample(self, num_samples: int, y) -> torch.Tensor:
#         # y has shape (m, d)
#         # return shape (num_samples, m, d) with y as the values being conditioned on (i.e., mean in this case)
#         assert len(y.shape) == 2 and y.shape[1] == self.dim
#         return torch.randn((num_samples, y.shape[0], y.shape[1])) * self.std + y
#
#     @checker
#     def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         # x is of shape (N,d), y is of shape (M,d)
#         # return shape (N,M)
#         x = x.unsqueeze(1)
#         y = y.unsqueeze(0)
#         return -0.5 * (
#                 torch.sum(((x - y) / self.std) ** 2, dim=2) + torch.log(2 * torch.pi * self.std * self.std).sum())
#
#
# # define a potential function that we want to sample from
# def potential(z: torch.Tensor) -> torch.Tensor:
#     z1, z2 = z[:, 0], z[:, 1]
#     w1 = torch.sin(2 * pi * z1 / 4)
#     return -0.5 * ((z2 - w1) / 0.4) ** 2
#
#
# # sample by mh_sampling
# num_samples, num_chain = 50000, 3  # the number of independent MC chains
# samples, _ = mh_sampling(num_samples, target=potential,
#                          transit=ConditionalMultiGauss(torch.ones(2)),
#                          initial=torch.ones((num_chain, 2)),
#                          burn_in=5000)
# # samples of shape (num_samples, num_chain, 2)
# # Let's plot the samples of the first chain
# import matplotlib.pyplot as plt
#
# vis_samples = samples[:, 0, :]
# bound = 4
# plt.figure()
# plt.scatter(vis_samples[:, 0], vis_samples[:, 1], s=1)
# plt.title("Samples from the potential by MH sampling")
# plt.xlim(-bound, bound)
# plt.ylim(-bound, bound)
#
# # plot golden potential function
# x = torch.linspace(-bound, bound, 100)
# y = torch.linspace(-bound, bound, 100)
# xx, yy = torch.meshgrid(x, y)
# grid_data = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)
# value = potential(grid_data)
# plt.figure()
# plt.scatter(grid_data[:, 0], grid_data[:, 1], c=torch.exp(value), cmap='viridis')
# plt.title('golden potential function')
# plt.xlim(-bound, bound)
# plt.ylim(-bound, bound)
# plt.show()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sampler.model import RealNVP
from sklearn import datasets
from sampler.functional.loss import KLGenLoss
from sampler.distribution import TDWrapper
from torch.distributions import MultivariateNormal
class Feedforward(nn.Module):
    def __init__(self, hidden_dims):
        super(Feedforward, self).__init__()

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)])
        self.activation = nn.LeakyReLU(0.2)
    def forward(self, x):
        for i in range(len(self.hidden_layers) - 1):
            x = self.activation(self.hidden_layers[i](x))
        x = self.hidden_layers[-1](x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_trans = 12
module = RealNVP(dim=2,
                num_trans=num_trans,
                scale_net=nn.ModuleList(
                         [Feedforward([1, 128, 128, 128, 1]) for _ in
                          range(num_trans)]),
                shift_net=nn.ModuleList(
                         [Feedforward([1, 128, 128, 128, 1]) for _ in
                          range(num_trans)]),
                p_base=TDWrapper(MultivariateNormal(torch.zeros(2), torch.eye(2)))).to(device)
optimizer = torch.optim.Adam(module.parameters(), lr=0.0001)
num_steps = 1000
criterion = KLGenLoss()
for i in range(num_steps):
        z, _ = datasets.make_moons(n_samples=1000, noise=0.1)
        z = torch.Tensor(z).to(device)
        loss = criterion(module, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"iter {i}, loss: {loss.item()}")

# show the generated samples
samples, log_prob = module.sample(10000)
samples = samples.cpu().detach().numpy()
plt.figure()
plt.scatter(samples[:, 0], samples[:, 1])
plt.title("generated samples")
# show the golden dataset
plt.figure()
x, _ = datasets.make_moons(n_samples=1000, noise=0.1)
plt.scatter(x[:, 0], x[:, 1])
plt.title("real samples")
plt.show()

