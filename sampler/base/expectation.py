import numpy as np
import torch
from typing import Union, Tuple, Callable, Any, Optional, List
from .._common import Func, Distribution, Condistribution, BiProbTrans
from .base import mh_sampling
from .._utils import _get_params
from torch.distributions.categorical import Categorical
from .._common import _bpt_decorator


@_bpt_decorator
def importance_sampling(num_samples: int,
                        target: Union[Distribution, BiProbTrans, Func],
                        proposal: Union[Distribution, BiProbTrans],
                        eval_func: Optional[Func] = None,
                        resample_ratio: Optional[float] = 0.0,
                        ):
    r"""
    Importance sampling (IS) estimator to calculate the expectation of a function :math:`f(x)` with respect to a target distribution :math:`p(x)` using a proposal distribution :math:`q(x)`. Sampling-importance-Resampling (SIR) has been integrated and can be triggered to return samples.

    .. note:: IS works regardless of normalized or not. See Eq. (11.19) of [Bishop2006PRML]_ for the normalized case, and Eqs. (11.20)-(11.23) for how we handle the unnormalized case. SIR can be used to draw samples from the target distribution utilizing the importance weights. The argument ``resample_ratio`` controls the size of resamples. Note that our resampling process is always done without replacement. Because resampling with replacement is not always possible (unless Eq. (24.2) is True in [kimhung]_).

    .. math::

        \mathbb{E}[f(x)] = \int f(x) p(x) dx \approx \frac{1}{N} \sum_{i=1}^{N} w(x_i) f(x_i)

    where the weights are given by :math:`w(x) = \frac{p(x)}{q(x)}`. The provided ``eval_func`` can be a multi-dimensional function. See Section 11.1.5 of [Bishop2006PRML]_ and [kimhung]_ on SIR.

    Args:
        num_samples (int): the number of samples to be drawn.
        target (Union[Distribution, Func]): the target distribution. Since target doesn't need to have a sampling method, it can be a function.
        proposal (Distribution): the proposal distribution.
        eval_func (Optional[Func]): the function whose expectation to be evaluated if it is not None.
        resample_ratio (Optional[float]): perform Sampling-Importance-Resampling (SIR) and return samples if set to larger than 0.
    """
    if not isinstance(resample_ratio, float) or resample_ratio < 0 or resample_ratio > 1:
        raise ValueError(f"The resample_ratio must be a float in [0,1], but got {resample_ratio}.")

    samples = proposal.sample(num_samples)
    weights = torch.exp(target(samples) - proposal(samples))

    resample, expectation = None, None

    if resample_ratio:
        normalized_weights = weights / torch.sum(weights)
        index = Categorical(normalized_weights).sample((max(1, int(resample_ratio * num_samples)),))
        resample = samples[index]

    if eval_func is not None:
        evals = eval_func(samples)
        weights = weights.view(-1, *[1] * (evals.ndim - 1))
        if (not hasattr(target, 'mul_factor') or not hasattr(proposal, 'mul_factor')
                or (target.mul_factor is None or proposal.mul_factor is None)):
            expectation = (weights * evals).mean(0) / weights.mean(0)
            # feature: when weights happens to be all 0, divided by 0 will return NaN.
            # TODO: discuss what should be returned in this case.
        else:
            expectation = (weights * evals).mean(0)

    return expectation, resample


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
            logpt, logpb = target(current), base(current)
            weight = anneal_log_criterion(logpt, logpb, beta[n + 1]) - anneal_log_criterion(logpt, logpb, beta[n])
        else:
            current_transit = transit[n] if isinstance(transit, (tuple, list)) else transit

            new, _ = mh_sampling(1,
                                 lambda x: anneal_log_criterion(target(x), base(x), beta[n]),
                                 current_transit,
                                 current,
                                 burn_in)

            logpt, logpb = target(new), base(new)
            weight += anneal_log_criterion(logpt, logpb, beta[n + 1]) \
                      - anneal_log_criterion(logpt, logpb, beta[n])
            current = new

    evals = eval_func(current)
    weight = torch.exp(weight).view(-1, *tuple(range(1, evals.ndim)))
    return (weight * evals).mean(0) / weight.mean(0)


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
            obj = torch.mean(evals.detach() * module(samples).view(-1, *tuple(range(1, evals.ndim)))
                             + evals, dim=0)
            grad_input = torch.autograd.grad(obj, param, grad_output)
        return None, None, None, *grad_input


score_estimator = lambda num_sample, model, func: ScoreEstimator.apply(num_sample, model, func, *_get_params(model))
