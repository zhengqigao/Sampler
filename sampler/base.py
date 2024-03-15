import numpy as np
import torch
from typing import Union, Tuple, Callable, Any, Optional, List
from ._common import Func, Distribution, Condistribution
import warnings
from ._utils import _alias
from .distribution import MultivariateNormal

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
        return (weights * evals).mean(0) / (proposal.norm / target.norm)


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
                target: Union[Distribution, Callable],
                transit: Condistribution,
                initial: torch.Tensor,
                burn_in: Optional[int] = 0) -> Tuple[torch.Tensor, Any]:
    r"""
    Metropolis-Hastings (MH) sampling to draw samples from a target distribution using a proposal distribution. See Section 11.2.2. of [Bishop2006PRML]_.

    .. note:: `num_samples` represents the number of samples will be returned. `burn_in` represents the first `burn_in` samples will be discarded. Thus, there will be `num_samples + burn_in` samples in total, with the first one being the `initial` and all the remaining ones to be generated by MH.

    Args:
        num_samples (int): the number of samples to be returned.
        target (Union[Distribution, Callable]): the target distribution. It doesn't need to have a sampling function, so we allow it to be a callable as well.
        transit (Distribution): the transition distribution.
        initial (torch.Tensor): the initial point to start the sampling process. The first dimension is the batch dimension B, and (num_samples, B, ...) will be returned.
        burn_in (Optional[int]): the number of burn-in samples to be discarded, default to 0.
    """

    if burn_in < 0:
        raise ValueError(f"The number of burn-in samples should be non-negative, but got burn_in = {burn_in}.")
    if num_samples <= 0:
        raise ValueError(f"The number of samples to be drawn should be positive, but got num_samples = {num_samples}.")
    elif num_samples == 1:
        return initial, None

    if initial.ndim == 1:  # tolerate the user to feed in one chain, reshape to (1, D) when given (D,)
        initial = initial.view(1, -1)

    samples, num_accept = torch.clone(initial).unsqueeze(0), torch.zeros(initial.shape[0])

    while samples.shape[0] < num_samples + burn_in:
        new = transit.sample(1, y=initial).view(initial.shape)
        ratio = target(new, in_log=True) + transit(initial, new, in_log=True).diag() \
                - target(initial, in_log=True) - transit(new, initial, in_log=True).diag()
        ratio = torch.exp(ratio)  # min(1, ratio) is not necessary.
        accept = torch.rand(ratio.shape) <= ratio
        num_accept += accept
        new[~accept] = initial[~accept]
        samples = torch.cat([samples, new.unsqueeze(0)], dim=0)
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
    leave the intermediate distributions invariant, i.e., :math:`p_j(x') = \int T(x', x) p_{j}(x) dx`. To guratantee
    the invariances, the provided `transit` conditional distribution will be used to define a Metropolis-Hasting
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
        burin_in (Optional[int]): the number of burn-in samples to be discarded in Markov Chain, default to 3.
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


def langevin_monte_carlo(num_samples: int,
                         target: Distribution,
                         step_size: float,
                         initial: torch.Tensor,
                         adjusted: Optional[bool] = False,
                         burn_in: Optional[int] = 0) -> torch.Tensor:
    r"""
    Langevin Monte Carlo (LMC) to draw samples from a target distribution.

    Args:
        num_samples (int): the number of samples to be returned.
        target (Distribution): the target distribution.
        step_size (float): the step size to discretize the Langevin dynamics.
        adjusted (Optional[bool]): whether to adjust the acceptance ratio using the Metropolis-Hasting criterion, default to False.
        burn_in (Optional[int]): the number of burn-in samples to be discarded, default to 0.
    """

    if isinstance(num_samples, int) != True or num_samples <= 0:
        raise ValueError(
            f"The number of samples to be drawn should be a positive integer, but got num_samples = {num_samples}.")
    if isinstance(step_size, float) != True or step_size <= 0:
        raise ValueError(f"The step size should be positive, but got tau = {step_size}.")

    if initial.ndim == 1: # tolerate the user to feed in one chain, reshape to (1, D) when given (D,)
        current = initial.view(1, -1)
    else:
        current = initial

    samples = torch.clone(current.unsqueeze(0))

    current.requires_grad = True
    logp_current = target(current, in_log=True)
    # a trick to make the objective to be a scalar so that the gradient can be computed.
    log_grad_current = torch.autograd.grad(logp_current.sum(), current)[0]
    current.requires_grad = False

    while samples.shape[0] < num_samples + burn_in:
        noise = torch.randn_like(current)
        new = (current + step_size * log_grad_current + (2 * step_size) ** 0.5 * noise).detach()

        new.requires_grad = True
        logp_new = target(new, in_log=True)
        log_grad_new = torch.autograd.grad(logp_new.sum(), new)[0]
        new.requires_grad = False

        if adjusted:
            accept = torch.rand(new.shape[0]) <= torch.exp((logp_new - logp_current) + (
                - 0.5 * torch.sum((current - new - step_size * log_grad_new) ** 2, dim = tuple(range(1, current.ndim))) / (4 * step_size)
                + 0.5 * torch.sum((new - current - step_size * log_grad_current) ** 2, dim = tuple(range(1, current.ndim))) / (4 * step_size)))

            logp_new[~accept] = logp_current[~accept]
            log_grad_new[~accept] = log_grad_current[~accept]
            new[~accept] = current[~accept]

        current, logp_current, log_grad_current = new, logp_new, log_grad_new
        samples = torch.cat([samples, new.unsqueeze(0)], dim=0)

    return samples[burn_in:].detach()


def hamiltonian_monte_carlo(num_samples: int,
                            target: Distribution,
                            step_size: float,
                            num_leapfrog: int,
                            initial: torch.Tensor,
                            kinetic: Optional[Distribution] = None,
                            burn_in: Optional[int] = 0) -> torch.Tensor:
    r"""
    Hamiltonian Monte Carlo (HMC) to draw samples from a target distribution.

    Args:
        num_samples (int): the number of samples to be returned.
        target (Distribution): the target distribution.
        step_size (float): the step size to discretize the Hamiltonian dynamics.
        num_leapfrog (int): the number of leapfrog steps to be taken.
        initial (torch.Tensor): the initial point to start the sampling process.
        kinetic (Optional[Distribution]): the kinetic distribution, default to a standard multivariate Gaussian.
        burn_in (Optional[int]): the number of burn-in samples to be discarded, default to 0.

    """

    dim = target.sample(1).shape[1]
    if kinetic is None:
        kinetic = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    elif not isinstance(kinetic, Distribution):
        raise ValueError(f"The kinetic distribution should be an instance of Distribution, but got {type(kinetic)}.")
    elif kinetic.sample(1).shape[1] != dim:
        raise ValueError(
            f"The dimension of the sample drawn from the kinetic distribution should equal that from the target distribution, "
            f"but got {kinetic.sample(1).shape[1]} and {dim}.")

    if initial.ndim == 1:  # tolerate the user to feed in one chain, reshape to (1, D) when given (D,)
        initial.view(1, -1)

    samples = torch.clone(initial.unsqueeze(0))

    while samples.shape[0] < num_samples + burn_in:
        initial_q, initial_p = samples[-1], kinetic.sample(initial.shape[0])  # draw a new momentum
        current_q, current_p = initial_q, initial_p

        # make a half step for momentum at the beginning
        current_q.requires_grad = True
        logq_current = target(current_q, in_log=True)
        logq_grad_current = torch.autograd.grad(logq_current.sum(), current_q)[0]
        current_q.requires_grad = False
        current_p = current_p + 0.5 * step_size * logq_grad_current

        for iter in range(num_leapfrog):

            # full step for position
            current_p.requires_grad = True
            logp_current = kinetic(current_p, in_log=True)
            logp_grad_current = torch.autograd.grad(logp_current.sum(), current_p)[0]
            current_p.requires_grad = False
            current_q = current_q - step_size * logp_grad_current

            # full step for momentum, except for the last iteration
            current_q.requires_grad = True
            logq_current = target(current_q, in_log=True)
            logq_grad_current = torch.autograd.grad(logq_current.sum(), current_q)[0]
            current_q.requires_grad = False
            current_p = current_p + step_size * logq_grad_current * (1 if iter != num_leapfrog - 1 else 0.5)

        # negative momentum in the end
        current_p = -current_p

        # MH criterion to accept or reject the sample
        logH_new = kinetic(current_p, in_log=True) + logq_current
        logH_initial = kinetic(initial_p, in_log=True) + target(initial_q, in_log=True)
        accept = torch.rand(logH_new.shape) <= torch.exp(-logH_new + logH_initial)

        initial_q[accept] = current_q[accept]
        samples = torch.cat([samples, initial_q.unsqueeze(0)], dim=0)



    return samples[burn_in:].detach()