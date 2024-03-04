import numpy as np
import torch
from typing import Union, Tuple, Callable, Any, Optional
from ._common import Func, Distribution
import warnings

__all__ = ['importance_sampling', 'rejection_sampling', 'MH_sampling']


def importance_sampling(num_samples: int,
                        target: Distribution,
                        proposal: Distribution,
                        eval_func: Func,
                        ) -> float:
    r"""
    Importance sampling (IS) estimator to calculate the expectation of a function :math: `f(x)` with respect to a target distribution :math:`p(x)` using a proposal distribution :math:`q(x)`. The estimator is given by:

    .. math::

        \mathbb{E}[f(x)] = \int f(x) p(x) dx \approx \frac{1}{N} \sum_{i=1}^{N} w(x_i) f(x_i)

    where the weights are given by :math:`w(x) = \frac{p(x)}{q(x)}`. The provided ``eval_func`` can be a multi-dimensional function.

    .. note:: IS works regardless of normalized or not. See Eq. (11.19) of [Bishop2006PRML]_ for the normalized case, and Eqs. (11.20)-(11.23) for how we handle the unnormalized case.

    Args:
        num_samples (int): the number of samples to be drawn.
        target (Distribution): the target distribution.
        proposal (Distribution): the proposal distribution.
        eval_func (Func): the function to be evaluated.
    """

    samples = proposal.sample(num_samples)
    evals = eval_func(samples)
    weights = torch.exp(target(samples, in_log=True) - proposal(samples, in_log=True))
    weights = weights.view(-1, *tuple(range(1, evals.ndim)))

    if target.norm is None or proposal.norm is None:
        return (weights * evals).mean(0) / weights.mean(0)
    else:
        return (weights * evals).mean(0)


def rejection_sampling(num_samples: int, target: Distribution, proposal: Distribution, k: float) -> Tuple[
    torch.Tensor, Any]:
    r"""
    Rejection sampling to draw samples from a target distribution using a proposal distribution and a scaling factor :math: `k>0`. See Section 11.1.2 of [Bishop2006PRML]_.

    Args:
        num_samples (int): the number of samples to be drawn.
        target (Distribution): the target distribution.
        proposal (Distribution): the proposal distribution.
        k (float): a positive constant such that :math: `k q(x) \geq \tilde{p}(x)`.
    """

    if k <= 0:
        raise ValueError(f"The scaling factor k should be positive, but got k = {k}.")

    # TODO: perform the comparison in the log domain? Is it worthy to implement the squeezing function as in [Gilks1992ars]_?
    total_num_sample, reject_num_sample, accept_sample = 0, 0, None
    while (total_num_sample - reject_num_sample) < num_samples:
        samples = proposal.sample((num_samples - accept_sample.shape[0]) if accept_sample is not None else num_samples)
        evals = target(samples, in_log=False)
        bound = k * proposal(samples, in_log=False)
        if torch.any(bound < evals):
            raise ValueError(f"The scaling factor k = {k} is not large enough.")
        u = torch.rand_like(bound) * bound
        current_accept_samples = samples[evals > u]
        if accept_sample is None:
            accept_sample = current_accept_samples
        else:
            accept_sample = torch.cat([accept_sample, current_accept_samples], dim=0)
        reject_num_sample += torch.sum(evals <= u).item()
        total_num_sample += samples.shape[0]
    return accept_sample[:num_samples], {'rejection_rate': reject_num_sample / total_num_sample}


def adaptive_rejection_sampling():
    pass


def MH_sampling(num_samples: int, target: Distribution, proposal: Distribution, initial: torch.Tensor) -> Tuple[
    torch.Tensor, Any]:
    r"""
    Metropolis-Hastings (MH) sampling to draw samples from a target distribution using a proposal distribution.


    Args:
        num_samples (int): the number of samples to be drawn.
        target (Distribution): the target distribution.
        proposal (Distribution): the proposal distribution.
    """

    initial = initial.view(1, -1)
    samples, num_accept = torch.clone(initial), 0

    while samples.shape[0] < num_samples:
        new = proposal.sample(1, y=initial)
        ratio = target(new, in_log=True) + proposal(initial, new, in_log=True) \
                - target(initial, in_log=True) - proposal(new, initial, in_log=True)
        ratio = min(1, torch.exp(ratio).item())
        if torch.rand(1) <= ratio:
            samples = torch.cat([samples, new], dim=0)
            num_accept, initial = num_accept + 1, new
        else:
            samples = torch.cat([samples, initial], dim=0)
    return samples, {'acceptance_rate': num_accept / (samples.shape[0] - 1)}
