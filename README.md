# Sampler

## Introduction

This library is dedicated to providing a comprehensive collection of sampling methods, including but not limited to classical sampling techniques, deep learning-based samplers, and rare event samplers. It is designed with PyTorch data formats in mind for seamless integration into modern machine learning workflows. **It is under active development and will be updated daily.** 

## Quick Example

```python
import torch
from torch.distributions import MultivariateNormal
from sampler import importance_sampling, Wrapper

# define the target and proposal distributions using the Wrapper class
target = Wrapper(MultivariateNormal(torch.Tensor([-1,1,0.5]), torch.eye(3)))
proposal = Wrapper(MultivariateNormal(torch.zeros(3), torch.eye(3)))

# define a function to estimate the expectation of
f = lambda x: x

# use proposal to estimate the expectation of f with respect to the target
results = importance_sampling(10000, target, proposal, f)
print("Test mean:", results)

```

## List of Algorithms

This section lists the algorithms that have been implemented and the algorithms that will be implemented in the future. This section will be updated regularly as new algorithms are added. We use :cat:, :dog:, and :rabbit: to represent classical, deep learning-based, and rare event samplers, respectively, because cat, dog, and rabbit have the same first letter as classical, deep learning-based, and rare event samplers, respectively. Note that this categories might be updated along development.


The following algorithms have been implemented:

- [Importance sampling](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) :cat:
- [Rejection sampling](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) :cat:
- [Metropolis-Hastings sampling](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) :cat:
- [Gibbs sampling](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) :cat:

The following algorithms will be implemented:

- [Annealed importance sampling](https://arxiv.org/abs/physics/9803008) :cat:
- 
