# Sampler

## Introduction

This library is dedicated to providing a comprehensive collection of sampling methods, including but not limited to classical sampling techniques, deep learning-based samplers (specifically, normalizing flows). It is designed with PyTorch data formats in mind for seamless integration into modern machine learning workflows. 

**This project is under active development and will be updated daily.** At this point, the implemented functions are in Beta stage. Things (functions, code structures, method arguments, class definitions) might be subjected to change without notice. If you are interested in using the library, we suggest to take a try and check back later for updates. Also, please feel free to raise an issue for any bugs or feature requests. We will gradually add tutorials and documentations for better user experience. 

## Installation

When we believe the library is stable enough, we will release it on pypi and conda. For now, installing the library can be done with the `pip` command.

```bash

pip install git+https://github.com/zhengqigao/Sampler.git

```

## List of Algorithms and Models

This section lists the algorithms and models that have been implemented and that will be implemented in the future. This section will be updated regularly as new algorithms and models are added. We use :cat: and :dog: to represent classical and deep learning-related, respectively, noticing the first characters are c and d. **We use :tea: to represent the algorithms that have been tested.** The following algorithms have been implemented:

- [Importance sampling](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf), [Sampling Importance Resampling](https://onlinelibrary.wiley.com/doi/pdf/10.1002/0470090456.ch24#:~:text=He%20called%20it%20the%20sampling,of%20size%20m%20as%20output.) :cat:
- [Rejection sampling](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) :cat:
- [Metropolis-Hastings sampling](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) :cat:, :tea:
- [Gibbs sampling](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) :cat:
- [Annealed importance sampling](https://arxiv.org/abs/physics/9803008) :cat:
- [Langevin Monte Carlo](https://abdulfatir.com/blog/2020/Langevin-Monte-Carlo/) :cat:
- [Hamiltonian Monte Carlo](https://arxiv.org/pdf/1206.1901.pdf) :cat:
- [Score Estimator](http://stillbreeze.github.io/REINFORCE-vs-Reparameterization-trick/) :cat:
- [Affine Coupling Flow](https://arxiv.org/abs/1605.08803), [RealNVP](https://arxiv.org/abs/1605.08803), [NICE](https://arxiv.org/pdf/1410.8516) :dog:
- [Planar Flow](https://arxiv.org/pdf/1505.05770) :dog:
- [Radial Flow](https://arxiv.org/pdf/1505.05770) :dog:
- [Glow: Generative Flow with Invertible 1×1 Convolutions](https://arxiv.org/abs/1807.03039) :dog:
- [MADE: Masked Autoencoder Distribution Estimator](https://arxiv.org/pdf/1502.03509) :dog:
- [MAF: Masked Autoregressive Flow](https://arxiv.org/pdf/1705.07057) :dog:



## Quick Example

Our main effort now is on  developing the code, tutorials and documentations will be added in the near future. At this point, checking the scripts under the test folder or directly reading the docstring of a method/class is the most straightforward way to understand how to use the library. Here we show a few example usages of the library. 

### 1. Importance Sampling and Basics of Defining a Distribution Class

```python
import torch
from torch.distributions import MultivariateNormal
from sampler.base import importance_sampling
from sampler import Distribution, checker
from sampler.distribution import TDWrapper

# define the target and proposal distributions using the Wrapper
target = TDWrapper(MultivariateNormal(torch.Tensor([-1, 1, 0.5]), torch.eye(3)))
proposal = TDWrapper(MultivariateNormal(torch.zeros(3), torch.eye(3)))

# define a function to estimate the expectation of. It can be a multidimensional function.
# In this example, we consider a R^3 to R^3 identity function.
f = lambda x: x

# use proposal to estimate the expectation of f with respect to the target
# As in this case f(x)=x, we essentially estimate the mean of the target distribution.
results, _ = importance_sampling(10000, target, proposal, f)
print("Test mean:", results)

# Or we can directly define a distribution class in Sampler library.
# The 'sample' and 'log_prob' methods need to be implemented by users.
# It is suggested to add the checker decorator to enable automatic checking.
class MultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)
        self.std = std if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)
        self.dim = len(self.mean)
        self.mul_factor = 1.0 # the normalization factor, when the distribution is correctly normalized, it should be 1.0

    @checker
    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    @checker
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
                torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
                + torch.log(2 * torch.pi * self.std * self.std).sum()
        )

target = MultiGauss([-1, 1, 0.5], [1,1,1])
proposal = MultiGauss([0, 0, 0], [1,1,1])
results, _ = importance_sampling(10000, target, proposal, f)
print("Test mean:", results)
```

### 2. Metropolis-Hastings Sampling

```python

import torch
from sampler import Condistribution, checker
from sampler.base import mh_sampling
from math import pi


# define a conditional distribution class in Sampler library.
# The 'sample' and 'log_prob' methods need to be implemented by users.
# It is suggested to add the checker decorator to enable automatic checking.
class ConditionalMultiGauss(Condistribution):
    def __init__(self, std):
        super().__init__()
        self.std = std if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)
        self.dim = len(std)

    @checker
    def sample(self, num_samples: int, y) -> torch.Tensor:
        # y has shape (m, d)
        # return shape (num_samples, m, d) with y as the values being conditioned on (i.e., mean in this case)
        assert len(y.shape) == 2 and y.shape[1] == self.dim
        return torch.randn((num_samples, y.shape[0], y.shape[1])) * self.std + y

    @checker
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x is of shape (N,d), y is of shape (M,d)
        # return shape (N,M)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return -0.5 * (
                torch.sum(((x - y) / self.std) ** 2, dim=2) 
                + torch.log(2 * torch.pi * self.std * self.std).sum())


# define a potential function that we want to sample from
# Here potential(z) means logp(z), i.e., p(z) = exp(potential(z)) upon to a normalization factor
def potential(z: torch.Tensor) -> torch.Tensor:
    z1, z2 = z[:, 0], z[:, 1]
    w1 = torch.sin(2 * pi * z1 / 4)
    return -0.5 * ((z2 - w1) / 0.4) ** 2


# sample by mh_sampling
num_samples, num_chain = 50000, 3  # the number of independent MC chains
samples, _ = mh_sampling(num_samples, target=potential,
                         transit=ConditionalMultiGauss(torch.ones(2)),
                         initial=torch.ones((num_chain, 2)),
                         burn_in=5000)

# samples of shape (num_samples, num_chain, 2)
# Let's plot the samples of the first chain
import matplotlib.pyplot as plt

vis_samples = samples[:, 0, :]
bound = 4
plt.figure()
plt.scatter(vis_samples[:, 0], vis_samples[:, 1], s=1)
plt.title("Samples from the potential by MH sampling")
plt.xlim(-bound, bound)
plt.ylim(-bound, bound)

# plot golden potential function
x = torch.linspace(-bound, bound, 100)
y = torch.linspace(-bound, bound, 100)
xx, yy = torch.meshgrid(x, y)
grid_data = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)
value = potential(grid_data)
plt.figure()
plt.scatter(grid_data[:, 0], grid_data[:, 1], c=torch.exp(value), cmap='viridis')
plt.title('golden potential function')
plt.xlim(-bound, bound)
plt.ylim(-bound, bound)
plt.show()

```

### 3. Normalizing Flows for Generation Tasks

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sampler.model import RealNVP
from sklearn import datasets
from sampler.functional.loss import KLGenLoss
from sampler.distribution import TDWrapper
from torch.distributions import MultivariateNormal

# define a feedforward network for the scale and shift functions in RealNVP
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
p_base=TDWrapper(MultivariateNormal(torch.zeros(2), torch.eye(2)))
module = RealNVP(dim=2,
                num_trans=num_trans,
                scale_net=nn.ModuleList(
                         [Feedforward([1, 128, 128, 128, 1]) for _ in
                          range(num_trans)]),
                shift_net=nn.ModuleList(
                         [Feedforward([1, 128, 128, 128, 1]) for _ in
                          range(num_trans)]),
                p_base=p_base).to(device)
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

```

### 4. Normalizing Flows for Density Estimation Tasks

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sampler.model import RealNVP
from sampler.functional.loss import KLDenLoss, ScoreDenLoss
from sampler.distribution import TDWrapper
from torch.distributions import MultivariateNormal
from math import pi
import numpy as np

# define a feedforward network for the scale and shift functions in RealNVP
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

# define a potential function that we want the flow model to learn
# Here potential(z) means logp(z)
def potential(z: torch.Tensor) -> torch.Tensor:
    z1, z2 = z[:, 0], z[:, 1]
    w1 = torch.sin(2 * np.pi * z1 / 4)
    return -0.5 * ((z2 - w1) / 0.4) ** 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_trans = 12
p_base = TDWrapper(MultivariateNormal(torch.zeros(2), torch.eye(2)))
module = RealNVP(dim=2,
                num_trans=num_trans,
                scale_net=nn.ModuleList(
                         [Feedforward([1, 128, 128, 128, 1]) for _ in
                          range(num_trans)]),
                shift_net=nn.ModuleList(
                         [Feedforward([1, 128, 128, 128, 1]) for _ in
                          range(num_trans)]),
                p_base=p_base).to(device)

optimizer = torch.optim.Adam(module.parameters(), lr=0.0001)
max_iter = 500
loss_list = []
batch_size = 1000
criterion1 = KLDenLoss(log_p = potential)
criterion2 = ScoreDenLoss(log_p = potential)

for i in range(max_iter):
    loss1 = criterion1(module, batch_size)
    loss2 = criterion2(module, batch_size)
    loss_list.append(loss1.item())
    if torch.isnan(loss1).any() or torch.isinf(loss1).any() or i == max_iter - 1:
        plt.figure()
        plt.plot(loss_list)
        plt.title("Loss")
        break
    optimizer.zero_grad()
    loss1.backward() # loss2.backward() uses score estimator, and usually has large variance. It is recommended to use loss1.backward()
    optimizer.step()
    print(f"iter {i}, KLDenLoss: {loss1.item():.3f}, ScoreDenLoss: {loss2.item():.3f}")

# do visualization
bound = 4

plt.figure()
samples, log_prob = module.sample(50000)
samples = samples.detach().cpu().numpy()
log_prob = log_prob.detach().cpu().numpy()
plt.scatter(samples[:, 0], samples[:, 1], c=np.exp(log_prob).reshape(-1),
                cmap='viridis')
plt.colorbar()
plt.title('learnt module samples')
plt.xlim(-bound, bound)
plt.ylim(-bound, bound)



xx, yy = torch.meshgrid(torch.linspace(-bound, bound, 100), torch.linspace(-bound, bound, 100))
grid_data = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)
plt.figure()
plt.scatter(grid_data[:, 0], grid_data[:, 1], c=torch.exp(potential(grid_data)), cmap='viridis')
plt.title('golden result')
plt.show()
```