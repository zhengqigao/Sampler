import numpy as np
import torch
from typing import Union, Tuple, Callable, Any, Optional, List
from ._common import Func, Distribution, Condistribution
import warnings
from ._utils import _alias

__all__ = ['importance_sampling', 'rejection_sampling', 'mh_sampling', 'gibbs_sampling']


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
    Rejection sampling to draw samples from a target distribution using a proposal distribution and a scaling factor :math:`k>0`. See Section 11.1.2 of [Bishop2006PRML]_.

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


def mh_sampling(num_samples: int,
                target: Distribution,
                proposal: Condistribution,
                initial: torch.Tensor,
                burn_in: Optional[int] = 0) -> Tuple[torch.Tensor, Any]:
    r"""
    Metropolis-Hastings (MH) sampling to draw samples from a target distribution using a proposal distribution. See Section 11.2.2. of [Bishop2006PRML]_.


    Args:
        num_samples (int): the number of samples to be drawn.
        target (Distribution): the target distribution.
        proposal (Distribution): the proposal distribution.
        initial (torch.Tensor): the initial point to start the sampling process.
        burn_in (Optional[int]): the number of burn-in samples to be discarded, default to 0.
    """

    ## TODO: is there a batched version of MH sampling? Every time only one new sample is generated, which is not efficient.
    if burn_in < 0:
        raise ValueError(f"The number of burn-in samples should be non-negative, but got burn_in = {burn_in}.")

    initial = initial.view(1, -1)
    samples, num_accept = torch.clone(initial), 0

    while samples.shape[0] < num_samples + burn_in:
        new = proposal.sample(1, y=initial).view(initial.shape)
        ratio = target(new, in_log=True) + proposal(initial, new, in_log=True) \
                - target(initial, in_log=True) - proposal(new, initial, in_log=True)
        ratio = min(1, torch.exp(ratio).item())
        if torch.rand(1) <= ratio:
            samples = torch.cat([samples, new], dim=0)
            num_accept, initial = num_accept + 1, new
        else:
            samples = torch.cat([samples, initial], dim=0)
    return samples[burn_in:], {'acceptance_rate': num_accept / (samples.shape[0] - 1)}


def gibbs_sampling(num_samples: int,
                   condis: Union[Tuple[Condistribution], List[Condistribution], Condistribution],
                   initial: torch.Tensor,
                   burn_in: Optional[int] = 0) -> Tuple[torch.Tensor, Any]:
    r"""
    Gibbs sampling to draw samples given conditional distributions. See Section 11.3 of [Bishop2006PRML]_.


    .. note:: Even though given all those conditional probabilities, it is still impossible to directly evaluate the joint distribution density. However, Gibbs sampling provides a way to draw samples from the joint distribution.


    Args:
        num_samples (int): the number of samples to be drawn.
        condis (Union[Tuple[Condistribution], List[Condistribution], Condistribution]): the conditional distributions.
        initial (torch.Tensor): the initial point to start the sampling process.
        burn_in (Optional[int]): the number of burn-in samples to be discarded, default to 0.
    """

    if burn_in < 0:
        raise ValueError(f"The number of burn-in samples should be non-negative, but got burn_in = {burn_in}.")

    initial = initial.view(1, -1)
    dim = initial.shape[1]

    samples = torch.clone(initial)
    mask = torch.ones(dim, dtype=torch.bool)
    for i in range(num_samples + burn_in):
        for j in range(dim):
            mask[j] = False
            if isinstance(condis, (tuple, list)):
                initial[j] = condis[j].sample(1, y=samples[i][mask]).view(1, -1)
            elif isinstance(condis, Condistribution):
                initial[j] = condis.sample(1, y=samples[i][mask]).view(1, -1)
            else:
                raise ValueError(
                f"The conditional distributions should be a tuple, list or a single instance of Condistribution, but got {type(condis)}.")
            mask[j] = True
        samples = torch.cat([samples, initial], dim=0)
    return samples[burn_in:], None


def annealed_importance_sampling():
    pass