import warnings
import numpy as np
import torch
import math
from typing import Union, Tuple, Callable, Any, Optional, List
from .._common import _bpt_decorator, Func, Distribution, Condistribution, BiProbTrans
from torch.distributions import MultivariateNormal
from sampler._utils import LinearEnvelop1D

@_bpt_decorator
def rejection_sampling(num_samples: int,
                       target: Union[Distribution, BiProbTrans, Func],
                       proposal: Union[Distribution, BiProbTrans],
                       k: float
                       ) -> Tuple[torch.Tensor, Any]:
    r"""
    Rejection sampling to draw samples from a target distribution using a proposal distribution and a scaling factor :math:`k>0`. See Section 11.1.2 of [Bishop2006PRML]_.

    Args:
        num_samples (int): the number of samples to be drawn.
        target (Distribution): the target distribution.
        proposal (Distribution): the proposal distribution.
        k (float): a positive constant such that :math: `k q(x) \geq \tilde{p}(x)` holds for all `x`.
    """

    if k <= 0 or not math.isfinite(k) or math.isnan(k):
        raise ValueError(f"The scaling factor k should be a positive finite scalar, but got k = {k}.")
    
    # TODO: Is it worthy to implement the squeezing function as in [Gilks1992ars]_?
    total_num_sample, reject_num_sample, accept_sample = 0, 0, None
    while (total_num_sample - reject_num_sample) < num_samples:
        samples = proposal.sample((num_samples - accept_sample.shape[0]) if accept_sample is not None else num_samples)
        evals = torch.exp(target(samples))
        bound = k * torch.exp(proposal(samples))
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


@_bpt_decorator
def adaptive_rejection_sampling(num_samples: int,
                                target: Union[Distribution, BiProbTrans, Func],
                                lower: float,
                                upper: float,) -> Tuple[torch.Tensor, Any]:
    r"""
    Adaptive Rejection sampling to draw samples from a log-concave target distribution. See Section 11.1.3 of [Bishop2006PRML]_.

    Args:
        num_samples (int): the number of samples to be drawn.
        target (Distribution): the target distribution.
        lower (float): the lower point to start the grid, the derivate here should be positive.
        upper (float): the upper point to end the grid, the derivate here should be negative.
    """

    if lower > upper:
        raise ValueError(f"The value of lower point should be smaller than the upper point.")
    if np.isinf(lower) or np.isinf(upper):
        raise ValueError(f"The value of lower point or upper point is invalid.")

    eval_points = torch.Tensor([[lower], [upper]])

    eval_points.require_grad = True
    eval_bound = target(eval_points)
    '''
    log_grad_current = torch.autograd.grad(eval_bound.sum(), eval_points)[0]
    eval_points.requires_grad = False
    '''

    derivate_step = 1e-6 * (upper - lower)
    derivate_eval_points = torch.Tensor([[lower], [lower+derivate_step], [upper-derivate_step], [upper]])
    derivate_eval_bound = target(derivate_eval_points)
    derivate_lower = (derivate_eval_bound[1]-derivate_eval_bound[0])/derivate_step
    derivate_upper = (derivate_eval_bound[3] - derivate_eval_bound[2])/derivate_step
    if np.sign(derivate_lower) < 0:
        raise ValueError(f"The derivate at lower point is negative.")
    if np.sign(derivate_upper) > 0:
        raise ValueError(f"The derivate at upper point is positive.")
    log_grad_current = torch.cat((derivate_lower.reshape(1,1), derivate_upper.reshape(1,1)),dim=1).tolist()[0]

    if np.sign(log_grad_current[0]) < 0:
        raise ValueError(f"The derivate at lower point is negative.")
    if np.sign(log_grad_current[1]) > 0:
        raise ValueError(f"The derivate at upper point is positive.")
    grid = [[lower], [upper]]

    proposal = LinearEnvelop1D(grid, log_grad_current, eval_bound)

    # TODO: add abscissae of rejected points into the linear envolope distirbution, multivariate
    total_num_sample, reject_num_sample, accept_sample = 0, 0, None
    while (total_num_sample - reject_num_sample) < num_samples:
        # print("accept_sample.shape[0]: {}".format((num_samples - accept_sample.shape[0]) if accept_sample is not None else num_samples))
        samples = LinearEnvelop1D.sample((num_samples - accept_sample.shape[0]) if accept_sample is not None else num_samples)
        evals = target(samples, in_log=False)
        bound = proposal(samples, in_log=False)
        if torch.any(bound < evals):
            raise ValueError(f"Wrong envelop distribution.")
        u = torch.rand_like(bound) * bound
        current_accept_samples = samples[evals > u]
        if accept_sample is None:
            accept_sample = current_accept_samples
        else:
            accept_sample = torch.cat([accept_sample, current_accept_samples], dim=0)
        reject_num_sample += torch.sum(evals <= u).item()
        total_num_sample += samples.shape[0]
    return accept_sample[:num_samples], {'rejection_rate': reject_num_sample / total_num_sample}


def mh_sampling(num_samples: int,
                target: Union[Distribution, Func],
                transit: Condistribution,
                initial: torch.Tensor,
                burn_in: Optional[int] = 0,
                event_func: Optional[Func] = lambda _: False) -> Tuple[torch.Tensor, Any]:
    r"""
    Metropolis-Hastings (MH) sampling to draw samples from a target distribution using a transition conditional distribution. See Section 11.2.2. of [Bishop2006PRML]_.

    .. note:: `num_samples` represents the number of samples returned. `burn_in` represents the first `burn_in` samples will be discarded. Thus, there will be `num_samples + burn_in` samples in total, with the first one being the provided `initial` and all the remaining ones to be generated by MH.

    Args:
        num_samples (int): the number of samples to be returned when event_func always return False during the sampling.
        target (Union[Distribution, Func]): the target distribution. Since the target doesn't need to have a sampling method, it can be a function.
        transit (Distribution): the transition distribution. Both a density and a sampling function are needed.
        initial (torch.Tensor): the initial point to start the sampling process. The first dimension is the batch dimension B, and (num_samples, B, ...) will be returned.
        burn_in (Optional[int]): the number of samples to be discarded at the beginning of MCMC, default to 0.
        event_func (Optional[Func]): when it returns True, the MH sampling will terminate immediately; default to a function that always returns False.
    """

    if not isinstance(burn_in, int) or burn_in < 0:
        raise ValueError(f"The number of burn_in samples should be a non-negative integer, but got burn_in = {burn_in}.")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError(f"The number of samples to be drawn should be a positive integer, but got num_samples = {num_samples}.")
    elif num_samples == 1:
        return initial, None

    if initial.ndim == 1:  # tolerate only one chain provided, reshape to (1, D) when given (D,)
        initial = initial.view(1, -1)
    device = initial.device
    samples, num_accept = torch.clone(initial).unsqueeze(0).to(device), torch.zeros(initial.shape[0]).to(device)

    while samples.shape[0] < num_samples + burn_in:
        new = transit.sample(1, y=initial).view(initial.shape)
        ratio = target(new) + transit(initial, new).diag() \
                - target(initial) - transit(new, initial).diag()
        ratio = torch.exp(ratio).to(device) # min(1, ratio) is not necessary
        accept = torch.rand(ratio.shape).to(device) <= ratio
        num_accept += accept
        new[~accept] = initial[~accept]
        initial = new
        samples = torch.cat([samples, new.unsqueeze(0)], dim=0)
        event_value = event_func(samples)
        if (isinstance(event_value, bool) or # provide some flexibility allowing torch.Tensor([True])
                (isinstance(event_value, torch.Tensor) and event_value.dtype == torch.bool and event_value.numel() == 1)):
            if event_value:
                if samples.shape[0] >= burn_in:
                    return samples[burn_in:], {'acceptance_rate': num_accept / (samples.shape[0] - 1)}
                else:
                    warnings.warn("event_func is triggered before burn_in stage")
                    return samples, {'acceptance_rate': num_accept / (samples.shape[0] - 1)}
        else:
            raise ValueError(f"event_func should return a single boolean value, but got {type(event_value)}.")
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



def langevin_monte_carlo(num_samples: int,
                         target: Distribution,
                         step_size: float,
                         initial: torch.Tensor,
                         adjusted: Optional[bool] = False,
                         burn_in: Optional[int] = 0,
                         event_func: Optional[Func] = lambda _: False) -> torch.Tensor:
    r"""
    Langevin Monte Carlo (LMC) to draw samples from a target distribution.

    Args:
        num_samples (int): the number of samples to be returned.
        target (Distribution): the target distribution.
        step_size (float): the step size to discretize the Langevin dynamics.
        adjusted (Optional[bool]): whether to adjust the acceptance ratio using the Metropolis-Hasting criterion, default to False.
        burn_in (Optional[int]): the number of burn-in samples to be discarded, default to 0.
        event_func (Optional[Func]): when it returns True, the LMC will terminate immediately; default to a function that always returns False.
    """

    # TODO: missing initial in docstring above

    if isinstance(num_samples, int) != True or num_samples <= 0:
        raise ValueError(
            f"The number of samples to be drawn should be a positive integer, but got num_samples = {num_samples}.")
    if isinstance(step_size, float) != True or step_size <= 0:
        raise ValueError(f"The step size should be positive, but got tau = {step_size}.")

    if initial.ndim == 1:  # tolerate the user to feed in one chain, reshape to (1, D) when given (D,)
        current = initial.view(1, -1)
    else:
        current = initial

    samples = torch.clone(current.unsqueeze(0))

    current.requires_grad = True
    logp_current = target(current)
    # a trick to make the objective to be a scalar so that the gradient can be computed.
    log_grad_current = torch.autograd.grad(logp_current.sum(), current)[0]
    current.requires_grad = False

    while samples.shape[0] < num_samples + burn_in:
        noise = torch.randn_like(current)
        new = (current + step_size * log_grad_current + (2 * step_size) ** 0.5 * noise).detach()

        new.requires_grad = True
        logp_new = target(new)
        log_grad_new = torch.autograd.grad(logp_new.sum(), new)[0]
        new.requires_grad = False

        if adjusted:
            accept = torch.rand(new.shape[0]) <= torch.exp((logp_new - logp_current) + (
                    - 0.5 * torch.sum((current - new - step_size * log_grad_new) ** 2,
                                      dim=tuple(range(1, current.ndim))) / (4 * step_size)
                    + 0.5 * torch.sum((new - current - step_size * log_grad_current) ** 2,
                                      dim=tuple(range(1, current.ndim))) / (4 * step_size)))

            logp_new[~accept] = logp_current[~accept]
            log_grad_new[~accept] = log_grad_current[~accept]
            new[~accept] = current[~accept]

        current, logp_current, log_grad_current = new, logp_new, log_grad_new
        samples = torch.cat([samples, new.unsqueeze(0)], dim=0)
        event_value = event_func(samples)
        if (isinstance(event_value, bool) or # provide some flexibility allowing torch.Tensor([True])
                (isinstance(event_value, torch.Tensor) and event_value.dtype == torch.bool and event_value.numel() == 1)):
            if event_value:
                if samples.shape[0] >= burn_in:
                    return samples[burn_in:]
                else:
                    warnings.warn("event_func is triggered before burn_in stage")
                    return samples
        else:
            raise ValueError(f"event_func should return a single boolean value, but got {type(event_value)}.")

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
    if initial.ndim == 1:  # tolerate the user to feed in one chain, reshape to (1, D) when given (D,)
        initial.view(1, -1)

    dim = initial.shape[1]

    if kinetic is None:
        kinetic = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    elif not isinstance(kinetic, Distribution):
        raise ValueError(f"The kinetic distribution should be an instance of Distribution, but got {type(kinetic)}.")
    elif kinetic.sample(1).shape[1] != dim:
        raise ValueError(
            f"The dimension of the sample drawn from the kinetic distribution should equal that from the target distribution, "
            f"but got {kinetic.sample(1).shape[1]} and {dim}.")

    if not isinstance(step_size, float) or step_size <= 0:
        raise ValueError(f"The step size should be positive, but got tau = {step_size}.")
    if not isinstance(num_leapfrog, int) or num_leapfrog <= 0:
        raise ValueError(
            f"The number of leapfrog steps should be a positive integer , but got num_leapfrog = {num_leapfrog}.")

    samples = torch.clone(initial.unsqueeze(0))

    while samples.shape[0] < num_samples + burn_in:
        initial_q, initial_p = samples[-1], kinetic.sample(initial.shape[0])  # draw a new momentum
        current_q, current_p = initial_q, initial_p

        # make a half step for momentum at the beginning
        current_q.requires_grad = True
        logq_current = target(current_q)
        logq_grad_current = torch.autograd.grad(logq_current.sum(), current_q)[0]
        current_q.requires_grad = False
        current_p = current_p + 0.5 * step_size * logq_grad_current

        for iter in range(num_leapfrog):
            # full step for position
            current_p.requires_grad = True
            logp_current = kinetic(current_p)
            logp_grad_current = torch.autograd.grad(logp_current.sum(), current_p)[0]
            current_p.requires_grad = False
            current_q = current_q - step_size * logp_grad_current

            # full step for momentum, except for the last iteration
            current_q.requires_grad = True
            logq_current = target(current_q)
            logq_grad_current = torch.autograd.grad(logq_current.sum(), current_q)[0]
            current_q.requires_grad = False
            current_p = current_p + step_size * logq_grad_current * (1 if iter != num_leapfrog - 1 else 0.5)

        # negative momentum in the end
        current_p = -current_p

        # MH criterion to accept or reject the sample
        logH_new = kinetic(current_p) + logq_current
        logH_initial = kinetic(initial_p) + target(initial_q)
        accept = torch.rand(logH_new.shape) <= torch.exp(-logH_new + logH_initial)

        initial_q[accept] = current_q[accept]
        samples = torch.cat([samples, initial_q.unsqueeze(0)], dim=0)

    return samples[burn_in:].detach()
