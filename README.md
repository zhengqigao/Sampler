# Sampler

## Introduction

This library is dedicated to providing a comprehensive collection of sampling methods, including but not limited to classical sampling techniques, deep learning-based samplers, and rare event samplers. It is designed with PyTorch data formats in mind for seamless integration into modern machine learning workflows. 

**This project is under active development and will be updated daily.** At this point, the implemented functions are in Beta stage. Things (functions, code structures, documentations, even the library name) might be subjected to change without notice. If you are interested in using the library, we suggest to take a try and check back later for updates. Also, please feel free to raise an issue for any bugs or feature requests. 

## Installation

When we believe the library is stable enough, we will release the first version on pypi and conda. For now, to install the package, the `pip` command is needed.

```bash

pip install git+https://github.com/zhengqigao/Sampler.git

```

## Quick Example

```python
import torch
from torch.distributions import MultivariateNormal
from sampler.base.base import importance_sampling
from sampler.distribution import Wrapper

# define the target and proposal distributions using the Wrapper class
target = Wrapper(MultivariateNormal(torch.Tensor([-1, 1, 0.5]), torch.eye(3)))
proposal = Wrapper(MultivariateNormal(torch.zeros(3), torch.eye(3)))

# define a function to estimate the expectation of. It can be a multi-dimensional function. In this example, we consider a R^3 to R^3 identity function.
f = lambda x: x

# use proposal to estimate the expectation of f with respect to the target
results = importance_sampling(10000, target, proposal, f)
print("Test mean:", results)

```
## Who should use this library?

1. If you are doing research related to classical sampling methods (e.g., developing new MCMC algorithms), our library provides a comprehensive collection of sampling methods, which are extensively tested. You ca directly call them without wasting time on baseline implementations.




## List of Algorithms and Models

This section lists the algorithms and models that have been implemented and that will be implemented in the future. This section will be updated regularly as new algorithms and models are added. We use :cat: and :dog: to represent classical and deep learning-related, respectively, noticing the first characters are c and d. **We use :tea: to represent the algorithms that have been tested.**


The following algorithms have been implemented:

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
- [MADE: Masked Autoencoder Distribution Estimator](https://arxiv.org/pdf/1502.03509)
- [MAF: Masked Autoregressive Flow](https://arxiv.org/pdf/1705.07057) :dog:


A list of TODOs are shown below. **Please add your name after it in parentheses if you are working on it.** Move to the list above when finished development, and meanwhile please add a reference which your implementation is based on. Note that some algorithms below should be combined to the previous implemented algorithms, and some algorithms might be able to call previous implemented algorithms as subroutines. 

- Sequential Importance Sampling, Sequential Monte Carlo
- Adaptive Rejection Sampling
- The No-U-Turn Sampler
- Blocked Gibbs Sampling (might be combined into Gibbs sampling)
- Monte Carlo, and Quasi-Monte Carlo

