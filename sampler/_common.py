from typing import TypeVar, Callable, Union, Optional, Tuple, List, Set
import torch
from abc import ABC, abstractmethod, ABCMeta
import math
import torch.nn as nn
from torch.distributions import Distribution as TorchDistribution
import copy

# __all__ = ['Func', 'Distribution', 'Condistribution', '_BaseDistribution']

Func = TypeVar('Func', bound=Callable[[Union[torch.Tensor]], Union[torch.Tensor, bool]])


def _sample_checker(func, cls_name):
    def _wrapp_conditional_sample(*args, **kwargs):

        samples = func(*args, **kwargs)
        num_samples_expected = kwargs['num_samples'] if 'num_samples' in kwargs.keys() else args[1]
        num_condis_samples = (kwargs['y'] if 'y' in kwargs.keys() else args[1]).shape[0]

        if not isinstance(samples, torch.Tensor):
            raise ValueError("The returned samples must be of type torch.Tensor.")
        elif samples.ndim < 3:
            raise ValueError(
                "The returned samples must be of shape (num_samples, y.shape[0], ...), with at least three dims.")
        elif samples.shape[0] != num_samples_expected or samples.shape[1] != num_condis_samples:
            raise ValueError(
                f"The shape of returned samples is ({samples.shape[0]}, {samples.shape[1]}, ...), but it should be (num_samples, y.shape[0], ...), i.e., ({num_samples_expected},{num_condis_samples}, ...).")
        return samples

    def _wrapp_uncondtional_sample(*args, **kwargs):
        samples = func(*args, **kwargs)
        num_samples_expected = kwargs['num_samples'] if 'num_samples' in kwargs.keys() else args[1]
        if not isinstance(samples, torch.Tensor):
            raise ValueError("The returned samples must be of type torch.Tensor.")
        elif samples.ndim < 2:
            raise ValueError("The returned samples must be of shape (num_samples, ...), with at least two dims.")
        elif samples.shape[0] != num_samples_expected:
            raise ValueError(
                f"The number of samples drawn is {samples.shape[0]}, but it should be {num_samples_expected}.")
        return samples

    if cls_name == "Condistribution":
        return _wrapp_conditional_sample
    elif cls_name == "Distribution":
        return _wrapp_uncondtional_sample
    else:
        return func


def _density_checker(func, cls_name):
    def _wrap_conditional_density(*args, **kwargs):
        x = kwargs['x'] if 'x' in kwargs.keys() else args[1]
        y = kwargs['y'] if 'y' in kwargs.keys() else args[2]
        num_density_expected = x.shape[0]
        num_condis_expected = y.shape[0]

        if x.ndim == 1 or y.ndim == 1:
            raise ValueError("The input x and y must be of shape (num_samples, ...), with at least two dims.")

        density = func(*args, **kwargs)

        if not isinstance(density, torch.Tensor):
            raise ValueError("The returned density must be of type torch.Tensor.")
        elif density.ndim >= 3 or density.shape[0] != num_density_expected or density.shape[1] != num_condis_expected:
            raise ValueError(
                f"The returned density must be of shape (x.shape[0], y.shape[0]), i.e., ({num_density_expected}, {num_condis_expected}), but got {tuple(density.shape)}.")
        return density

    def _wrap_uncondtional_density(*args, **kwargs):
        x = kwargs['x'] if 'x' in kwargs.keys() else args[1]
        num_density_expected = x.shape[0]

        if x.ndim == 1:
            raise ValueError("The input x must be of shape (num_samples, ...), with at least two dims.")

        density = func(*args, **kwargs)

        if not isinstance(density, torch.Tensor):
            raise ValueError("The returned density must be of type torch.Tensor.")
        elif density.ndim >= 2 or density.shape[0] != num_density_expected:
            raise ValueError(
                f"The returned density must be of shape (x.shape[0],), i.e., ({num_density_expected},), but got {tuple(density.shape)}.")
        return density

    if cls_name == "Condistribution":
        return _wrap_conditional_density
    elif cls_name == "Distribution":
        return _wrap_uncondtional_density
    else:
        return func


class _Meta(ABCMeta):
    def __new__(cls, name, bases, dct):
        if 'sample' in dct and hasattr(bases[0], 'sample'):
            base_cls_name = bases[0].__name__
            dct['sample'] = _sample_checker(dct['sample'], base_cls_name)
        if 'log_prob' in dct and hasattr(bases[0], 'log_prob'):
            base_cls_name = bases[0].__name__
            dct['log_prob'] = _density_checker(dct['log_prob'], base_cls_name)
        return super().__new__(cls, name, bases, dct)


class _BaseDistribution(nn.Module, metaclass=_Meta):

    def __init__(self):
        super().__init__()

    @property
    def mul_factor(self):
        if not hasattr(self, '_mul_factor'):
            raise AttributeError(
                "The normalization constant is not provided, "
                "Please set `self.mul_factor = None` or `self.div_factor = None` if unknown.")
        return self._mul_factor

    @mul_factor.setter
    def mul_factor(self, value):
        if value is None:
            self._mul_factor = None
        elif isinstance(value, (int, float)) and value > 0:
            self._mul_factor = float(value)
        else:
            raise ValueError(f"The mul_factor must be a positive scalar, but got {value}.")

    @property
    def div_factor(self):
        return (1.0 / self.mul_factor) if self.mul_factor is not None else None

    @div_factor.setter
    def div_factor(self, value):
        if value is None:
            self.mul_factor = None
        elif isinstance(value, (int, float)) and value > 0:
            self.mul_factor = 1.0 / float(value)
        else:
            raise ValueError(f"The div_factor must be a positive scalar, but got {value}.")

    def sample(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        Evaluate the probability distribution.

        """

        result = self.log_prob(*args, **kwargs)

        if self.mul_factor is not None:
            return result + math.log(self.mul_factor)
        else:
            return result


class Distribution(_BaseDistribution):
    r"""
    When defining a distribution using this template, the users must implement the following methods:

    - ``sample``: draw samples from the PDF.
    - ``log_prob``: evaluate the log density function at given points.

    """

    def __init__(self):
        super().__init__()

    def sample(self, num_samples: int) -> torch.Tensor:
        r"""
        Draw samples from the distribution. The samples should be of shape (num_samples, ...).

        Args:
            num_samples (int): the number of samples to be drawn.
        """

        raise NotImplementedError

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Evaluate the density function :math:`log\tilde{p}(x)` at given :math:`x`. The returned values should be of shape (x.shape[0],).

        Args:
            x (torch.Tensor): the point(s) at which to evaluate the density function.
        """

        raise NotImplementedError


class Condistribution(_BaseDistribution):
    r"""
    When defining a distribution using this template, the users must implement the following methods:

    - ``sample``: draw samples from the PDF.
    - ``log_prob``: evaluate the density function at given points.

    """

    def __init__(self):
        super().__init__()

    def sample(self, num_samples, y: torch.Tensor) -> torch.Tensor:
        r"""
        Draw samples from the distribution :math: `p(\cdot | y)`. The returned samples should be of shape (num_samples, y.shape[0], ...)

        Args:
            num_samples (int): the number of samples to be drawn.
            y (torch.Tensor): the parameters being conditioned on.
        """

        raise NotImplementedError

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Evaluate the density function :math:`log\tilde{p}(x|y)` at given :math:`x`. The returned values should be of shape (x.shape[0], y.shape[0]).

        Args:
            x (torch.Tensor): the point(s) at which to evaluate the potential function.
            y (torch.Tensor): the parameters being conditioned on.
        """

        raise NotImplementedError


class InvProbTrans(nn.Module):
    r"""
    The base class for an invertible probabilistic transform. The forward and backward function must be implemented by
    the users, and satisfy the following relationship: x, 0 = model.backward(model.forward(x, 0)).

    """

    def __init__(self, p_base: Optional[Union[TorchDistribution, Distribution]] = None, *args, **kwargs):
        super().__init__()
        self.p_base = p_base

    def forward(self, x: torch.Tensor, log_prob: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[
        torch.Tensor, torch.Tensor]:
        r"""
        The forward transformation of the invertible probabilistic transform.

        Args:
            x (torch.Tensor): the input tensor.
            log_prob (Optional[Union[float, torch.Tensor]]): the log determinant of the Jacobian matrix before doing forward.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the transformed tensor and the log determinant of the Jacobian matrix.
        """
        raise NotImplementedError

    def backward(self, z: torch.Tensor, log_prob: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[
        torch.Tensor, torch.Tensor]:
        r"""
        The backward transformation of the invertible probabilistic transform.

        Args:
            z (torch.Tensor): the input tensor.
            log_prob (Optional[Union[float, torch.Tensor]]): the log determinant of the Jacobian matrix before doing backward.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the transformed tensor and the log determinant of the Jacobian matrix.
        """
        raise NotImplementedError

    def sample(self, num_samples: int, p_given: Optional[Union[TorchDistribution, Distribution]] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        r"""
        Draw samples from a base distribution.

        Args:
            num_samples (int): the number of samples to be drawn.
            p_given (Union[TorchDistribution, Distribution]): the distribution to be sampled from. If None, the base attribute of the instance will be used.
        """
        if p_given is None and self.p_base is None:
            raise ValueError("A base distribution is needed to do sampling. Please pass a distribution to p_given "
                             "or set the p_base attribute of the instance.")

        p = p_given if p_given is not None else self.p_base

        if isinstance(p, Distribution):
            samples = p.sample(num_samples)
            return self.forward(samples, p(samples))
        elif isinstance(p, TorchDistribution):
            samples = p.sample(torch.Size([num_samples]))
            return self.forward(samples, p.log_prob(samples))
        else:
            raise ValueError(f"The base distribution should be an instance of Distribution or TorchDistribution, "
                             f"but got {type(p_given)}.")

    def modify(self):
        self.ori_sample = self.sample
        self.ori_forward = self.forward

        def tmp_sample(inst, num_samples: int):
            return inst.ori_forward(inst.p_base.sample(num_samples), 0)[0]

        def tmp_forward(inst, z: torch.Tensor):
            x, log_det = self.backward(z, 0)
            return inst.p_base.log_prob(x) - log_det

        self.sample = tmp_sample.__get__(self)
        self.forward = tmp_forward.__get__(self)

    def restore(self):
        if hasattr(self, 'ori_sample') and hasattr(self, 'ori_forward'):
            self.sample = self.ori_sample
            self.forward = self.ori_forward
            del self.ori_sample, self.ori_forward


# def _wrapfunc_ipt(model_set: Set[InvProbTrans], func: Callable, *args, **kwargs):
#     original_methods = {}
#
#     for model in model_set:
#         # if model not in original_methods:  # The if statement is for the case when one model occurs multiple times in model_list
#         original_methods[model] = (model.sample, model.forward)
#
#         def tmp_sample(self, ori_forward: Callable):
#             return lambda num_samples: ori_forward(self.p_base.sample(num_samples), 0)[0]
#
#         def tmp_forward(self, z: torch.Tensor):
#             x, log_det = self.backward(z, 0)
#             return self.p_base.log_prob(x) - log_det
#
#         model.sample = tmp_sample(model, original_methods[model][1])
#         model.forward = tmp_forward.__get__(model)
#
#     # Execute the function with modified model methods
#     results = func(*args, **kwargs)
#
#     # Restore the original methods
#     for model in model_set:
#         model.sample, model.forward = original_methods[model]
#
#     return results


def _wrapfunc_ipt(model_set: Set[InvProbTrans], func: Callable, *args, **kwargs):
    for model in model_set:
        model.modify()

    results = func(*args, **kwargs)

    for model in model_set:
        model.restore()

    return results

def _ipt_decorator(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        model_list = set([arg for arg in args if isinstance(arg, InvProbTrans)] + \
                         [value for value in kwargs.values() if isinstance(value, InvProbTrans)])

        return func(*args, **kwargs) if len(model_list) == 0 else _wrapfunc_ipt(model_list, func, *args, **kwargs)

    return wrapper
