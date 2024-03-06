# Sampler

## Introduction

This library is dedicated to providing a comprehensive collection of sampling methods, including but not limited to classical sampling techniques, deep learning-based samplers, and rare event samplers. It is designed with PyTorch data formats in mind for seamless integration into modern machine learning workflows. 

**This project is under active development and will be updated daily.** At this point, the implemented functions are in Beta stage. Things (functions, code structures, documentations, even the library name) might be subjected to change without notice. If you are interested in using the library, we suggest to take a try and check back constantly for updates. Also, please feel free to raise an issue for any bugs or feature requests.

## Installation

To install the package, the `pip` command is needed.

```bash

pip install git+https://github.com/zhengqigao/Sampler.git

```

## Quick Example

```python
import torch
from torch.distributions import MultivariateNormal
from sampler.base import importance_sampling
from sampler.distribution import Wrapper

# define the target and proposal distributions using the Wrapper class
target = Wrapper(MultivariateNormal(torch.Tensor([-1, 1, 0.5]), torch.eye(3)))
proposal = Wrapper(MultivariateNormal(torch.zeros(3), torch.eye(3)))

# define a function to estimate the expectation of
f = lambda x: x

# use proposal to estimate the expectation of f with respect to the target
results = importance_sampling(10000, target, proposal, f)
print("Test mean:", results)

```

## List of Algorithms

This section lists the algorithms that have been implemented and the algorithms that will be implemented in the future. This section will be updated regularly as new algorithms are added. We use :cat:, :dog:, and :rabbit: to represent classical, deep learning-based, and rare event samplers, respectively, noticing the first characters are c, d, and r.


The following algorithms have been implemented:

- [Importance sampling](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) :cat:
- [Rejection sampling](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) :cat:
- [Metropolis-Hastings sampling](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) :cat:
- [Gibbs sampling](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) :cat:

The following algorithms will be implemented:

- [Annealed importance sampling](https://arxiv.org/abs/physics/9803008) :cat:

## Development TODOs:

- [zhengqi] Add sanity check from the developer side for conditional sampling function in the Condistribution class.