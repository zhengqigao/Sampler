import numpy as np
import torch
from typing import Union, Tuple, Callable, Any, Optional, List
from .._common import Func, Distribution, Condistribution
from .base import mh_sampling
from .._utils import _get_params
from torch.distributions.categorical import Categorical

def importance_sampling(num_samples: int,
                        target: Distribution,
                        proposal: Distribution,
                        eval_func: Func = None,
                        resampling: bool = False
                        ):
    r"""
    Importance sampling (IS) estimator to calculate the expectation of a function :math:`f(x)` with respect to a target distribution :math:`p(x)` using a proposal distribution :math:`q(x)`. The estimator is given by:

    .. note:: IS works regardless of normalized or not. See Eq. (11.19) of [Bishop2006PRML]_ for the normalized case, and Eqs. (11.20)-(11.23) for how we handle the unnormalized case.

    .. math::

        \mathbb{E}[f(x)] = \int f(x) p(x) dx \approx \frac{1}{N} \sum_{i=1}^{N} w(x_i) f(x_i)

    where the weights are given by :math:`w(x) = \frac{p(x)}{q(x)}`. The provided ``eval_func`` can be a multi-dimensional function.

    Args:
        num_samples (int): the number of samples to be drawn.
        target (Distribution): the target distribution.
        proposal (Distribution): the proposal distribution.
        eval_func (Func): the function to be evaluated.
        resampling (bool): the indicator of performing Sampling-Importance-Resampling(SIR).
    """
    samples = proposal.sample(num_samples)
    weights = torch.exp(target(samples, in_log=True) - proposal(samples, in_log=True))

    if resampling:
        nor_weights = weights/torch.sum(weights)
        discre_sampling = Categorical(weights)
        index = discre_sampling.sample(nor_weights.shape)
        resampling_list = torch.stack([samples[i] for i in index])

    if eval_func is not None:
        evals = eval_func(samples)
        weights = weights.view(-1, *tuple(range(1, evals.ndim)))
        if target.mul_factor is None or proposal.mul_factor is None:
            expectation = (weights * evals).mean(0) / weights.mean(0)
        else:
            expectation = (weights * evals).mean(0)

    if resampling and eval_func is not None:
        return expectation, resampling_list
    elif resampling:
        return resampling_list
    else:
        return expectation

def annealed_importance_sampling(num_samples: int,
                                 target: Distribution,
                                 base: Distribution,
                                 transit: Union[List[Condistribution], Tuple[Condistribution], Condistribution],
                                 eval_func: Func,
                                 beta: Union[Tuple[float], List[float]],
                                 anneal_log_criterion: Optional[Callable] = lambda logpt, logpb, beta: beta * logpt + (
                                         1 - beta) * logpb,
                                 burn_in: Optional[int] = 3) -> float:
    r"""
    Annealed importance sampling (AIS) estimator to calculate the expectation of a function :math: `f(x)` with
    respect to a target distribution :math:`p(x)` using a sequence of intermediate distributions :math:`p_N(x),
    p_{N-1}(x), ..., p_0(x)`.


    .. note:: The annealing sequence should be strictly increasing, its element represents the beta variable used in
    the anneal_log_criterion, corresponding to the weight/portion used on the target.  The transition distributions must
    leave the intermediate distributions invariant, i.e., :math:`p_j(x') = \int T(x', x) p_{j}(x) dx`. To guarantee
    the invariance, the provided `transit` conditional distribution will be used to define a Metropolis-Hasting
    transition. See Eq(11.44)- (11.45) of [Bishop2006PRML]_ for more details. By default, we use the annealing
    criterion: :math:`p_j(x) = p^{\beta_j}_{0}(x) p^{(1-\beta_j)}_{N}(x)`. See [Neal2001ais]_.


    Args:
        num_samples (int): the number of samples to be drawn.
        target (Distribution): the target distribution that the expectation to be estimated with respect to.
        base (Distribution): the base distribution.
        transit (Union[List[Condistribution], Tuple[Condistribution], Condistribution]): the transition distributions.
        eval_func (Func): the function to be evaluated.
        beta (Union[Tuple[float], List[float]]): the annealing sequence.
        anneal_log_criterion (Optional[Callable]): the annealed criterion, default to :math: `\beta \log p_t + (1-\beta) \log p_b`.
        burn_in (Optional[int]): the number of burn-in samples to be discarded in Markov Chain, default to 3.
    """

    if not isinstance(beta, (tuple, list)):
        raise ValueError(f"The annealing sequence should be a tuple or a list, but got {type(beta)}.")
    else:
        for i in range(len(beta) - 1):
            if beta[i] >= beta[i + 1]:
                raise ValueError(f"The annealing sequence should be strictly increasing.")
        if beta[0] != 0:
            beta = [0] + list(beta)
        if beta[-1] != 1:
            beta = list(beta) + [1]
        if len(beta) <= 2:
            raise ValueError(f"The length of the annealing sequence (including 0 and 1 at two extremes) "
                             f" should be larger than 3, but got {len(beta)}.")

    num_transit = len(beta) - 1

    if isinstance(transit, (tuple, list)) and len(transit) != num_transit - 1:
        raise ValueError(f"The number of transition distributions should equal "
                         f"the length of the annealing sequence, i.e.,"
                         f" {num_transit - 1}, but got {len(transit)}.")

    for n in range(num_transit):
        if n == 0:
            current = base.sample(num_samples)
            logpt, logpb = target(current, in_log=True), base(current, in_log=True)
            weight = anneal_log_criterion(logpt, logpb, beta[n + 1]) - anneal_log_criterion(logpt, logpb, beta[n])
        else:
            current_transit = transit[n] if isinstance(transit, (tuple, list)) else transit

            ## TODO: for the target argument in mh_sampling, inside MH sampling it might use in_log = True/False, but here here we restrict the annalead_log_criterion to be in log when using ais.
            new, _ = mh_sampling(1,
                                 lambda x, in_log=True: anneal_log_criterion(target(x, in_log=True),
                                                                             base(x, in_log=True), beta[n]),
                                 current_transit,
                                 current,
                                 burn_in)

            logpt, logpb = target(new, in_log=True), base(new, in_log=True)
            weight += anneal_log_criterion(logpt, logpb, beta[n + 1]) \
                      - anneal_log_criterion(logpt, logpb, beta[n])
            current = new

    evals = eval_func(current)
    weight = torch.exp(weight).view(-1, *tuple(range(1, evals.ndim)))
    return (weight * evals).mean(0) / weight.mean(0)

## TODO: add the other part derivative
class ScoreEstimator(torch.autograd.Function):
    r"""
    The REINFORCE algorithm, also known as the score function estimator, to estimate the gradient of the expectation of :math: `E_{p_{\theta}(x)}[f(x)]` with respect to the parameters of :math: `\theta`.
    """

    @staticmethod
    def forward(ctx, num_sample: int, module: torch.nn.Module, func: Func, *param: Tuple[torch.Tensor]) -> torch.Tensor:
        ctx.module = module
        samples = module.sample(num_sample)
        evals = func(samples)
        ctx.save_for_backward(samples, evals, *param)
        return torch.mean(evals, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        module = ctx.module
        samples, evals, *param = ctx.saved_tensors
        with torch.enable_grad():
            obj = torch.mean(evals.detach() * module(samples, in_log=True).view(-1, *tuple(range(1, evals.ndim)))
                             + evals, dim=0)
            grad_input = torch.autograd.grad(obj, param, grad_output)
        return None, None, None, *grad_input


score_estimator = lambda num_sample, model, func: ScoreEstimator.apply(num_sample, model, func, *_get_params(model))

