import numpy as np
import torch
from typing import Union, Tuple, Callable, Any
from common import _Func, Distribution


__all__ =['importance_sampling']


def importance_sampling(num_samples: int,
                        target: Distribution,
                        proposal: Distribution,
                        eval_func: _Func,) -> float:
    r"""
    Importance sampling estimator to calculate the expectation of a function :math: `f(x)` with respect to a target distribution :math:`p(x)`:

    .. math::

        \mathbb{E}[f(x)] = \int f(x) p(x) dx \approx \frac{1}{N} \sum_{i=1}^{N} w(x_i) f(x_i)

    where the weights are given by :math:`w(x) = \frac{p(x)}{q(x)}`.

    Args:
        num_samples (int): the number of samples to be drawn.
        target (PDF): the target distribution.
        proposal (PDF): the proposal distribution.
        eval_func (_Func): the function to be evaluated.

    """

    samples = proposal.sample(num_samples)
    evals = eval_func(samples)

    if type(samples) == np.ndarray:
        weights = np.exp(target(samples, in_log=True) - proposal(samples, in_log=True))
        weights = np.expand_dims(weights, tuple(range(1, evals.ndim)))
    elif type(samples) == torch.Tensor:
        weights = torch.exp(target(samples, in_log=True) - proposal(samples, in_log=True))
        weights = weights.view(-1, *tuple(range(1, evals.ndim)))
    else:
        raise ValueError(f"Samples must be a numpy array or a torch tensor, but got {format(type(samples))}")

    return (weights * evals).mean(0)

