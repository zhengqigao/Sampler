from typing import TypeVar, Callable, Union, Optional
import torch
from abc import ABC, abstractmethod, ABCMeta
import math
from torch.distributions import Distribution as TorchDistribution

# __all__ = ['Func', 'Distribution', 'Condistribution', '_BaseDistribution']

Func = TypeVar('Func', bound=Callable[[Union[torch.Tensor]], Union[torch.Tensor]])


def _sample_checker(func, cls_name):
    def _wrapp_conditional_sample(*args, **kwargs):

        samples = func(*args, **kwargs)
        num_samples_expected = kwargs['num_samples'] if 'num_samples' in kwargs.keys() else args[1]
        num_condis_samples = (kwargs['y'] if 'y' in kwargs.keys() else args[1]).shape[0]

        if not isinstance(samples, torch.Tensor):
            raise ValueError("The returned samples must be of type torch.Tensor.")
        elif samples.ndim < 3:
            raise ValueError("The returned samples must be of shape (num_samples, y.shape[0], ...), with at least three dims.")
        elif samples.shape[0] != num_samples_expected or samples.shape[1] != num_condis_samples:
            raise ValueError(f"The shape of returned samples is ({samples.shape[0]}, {samples.shape[1]}, ...), but it should be (num_samples, y.shape[0], ...), i.e., ({num_samples_expected},{num_condis_samples}, ...).")
        return samples
    def _wrapp_uncondtional_sample(*args, **kwargs):
        samples = func(*args, **kwargs)
        num_samples_expected = kwargs['num_samples'] if 'num_samples' in kwargs.keys() else args[1]
        if not isinstance(samples, torch.Tensor):
            raise ValueError("The returned samples must be of type torch.Tensor.")
        elif samples.ndim < 2:
            raise ValueError("The returned samples must be of shape (num_samples, ...), with at least two dims.")
        elif samples.shape[0] != num_samples_expected:
            raise ValueError(f"The number of samples drawn is {samples.shape[0]}, but it should be {num_samples_expected}.")
        return samples

    if cls_name == "Condistribution":
        return _wrapp_conditional_sample
    elif cls_name == "Distribution":
        return _wrapp_uncondtional_sample
    else:
        return func

def _density_checker(func, cls_name):
    def _wrapp_conditional_density(*args, **kwargs):
        density = func(*args, **kwargs)
        num_density_expected = (kwargs['x'] if 'x' in kwargs.keys() else args[1]).shape[0]
        num_condis_expected = (kwargs['y'] if 'y' in kwargs.keys() else args[2]).shape[0]
        if not isinstance(density, torch.Tensor):
            raise ValueError("The returned density must be of type torch.Tensor.")
        elif density.ndim >= 3 or density.shape[0] != num_density_expected or density.shape[1] != num_condis_expected:
            raise ValueError(f"The returned density must be of shape (x.shape[0], y.shape[0]), i.e., ({num_density_expected}, {num_condis_expected}), but got {tuple(density.shape)}.")
        return density
    def _wrapp_uncondtional_density(*args, **kwargs):
        density = func(*args, **kwargs)
        num_density_expected = (kwargs['x'] if 'x' in kwargs.keys() else args[1]).shape[0]
        if not isinstance(density, torch.Tensor):
            raise ValueError("The returned density must be of type torch.Tensor.")
        elif density.ndim >= 2 or density.shape[0] != num_density_expected:
            raise ValueError(f"The returned density must be of shape (x.shape[0],), i.e., ({num_density_expected},), but got {tuple(density.shape)}.")
        return density

    if cls_name == "Condistribution":
        return _wrapp_conditional_density
    elif cls_name == "Distribution":
        return _wrapp_uncondtional_density
    else:
        return func


class _Meta(ABCMeta):
    def __new__(cls, name, bases, dct):
        if 'sample' in dct and hasattr(bases[0], 'sample'):
            base_cls_name = bases[0].__name__
            dct['sample'] = _sample_checker(dct['sample'], base_cls_name)
        if 'evaluate_density' in dct and hasattr(bases[0], 'evaluate_density'):
            base_cls_name = bases[0].__name__
            dct['evaluate_density'] = _density_checker(dct['evaluate_density'], base_cls_name)
        return super().__new__(cls, name, bases, dct)



class _BaseDistribution(ABC, metaclass=_Meta):

    def __init__(self):
        self._norm = None

    @property
    def norm(self):
        if not hasattr(self, '_norm'):
            raise AttributeError(
                "The normalization constant is not provided, Please set `self.const = None` if unknown.")
        return self._norm

    @norm.setter
    def norm(self, value):
        if value is None:
            self._norm = None
        elif isinstance(value, (int, float)) and value > 0:
            self._norm = float(value)
        else:
            raise ValueError("The normalization constant must be a positive scalar.")

    @abstractmethod
    def sample(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def evaluate_density(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        r"""
        Evaluate the probability distribution.

        """

        result = self.evaluate_density(*args, **kwargs)
        if self.norm is not None:
            result = result * self.norm if not kwargs['in_log'] else result + math.log(self.norm)
        return result


class Distribution(_BaseDistribution):
    r"""
    Abstract class for probability density functions :math:`p(x)=c*\tilde{p}(x)`. When defining a distribution using this template, the users must implement the following methods:

    - ``sample``: draw samples from the PDF.
    - ``evaluate_density``: evaluate the density function at given points.

    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        r"""
        Draw samples from the distribution :math: `p(\cdot)`. The samples should be of shape (num_samples, ...).

        Args:
            num_samples (int): the number of samples to be drawn.
        """

        raise NotImplementedError

    @abstractmethod
    def evaluate_density(self, x: torch.Tensor, in_log: bool) -> torch.Tensor:
        r"""
        Evaluate the density function :math:`\tilde{p}(x)` at given :math:`x`. When in_log is True, the logarithm of the density function :math:`log\tilde{p}(x)` should be returned. The returned values should be of shape (x.shape[0],).

        Args:
            x (torch.Tensor): the point(s) at which to evaluate the density function.
            in_log (bool): the returned values are in natural logarithm scale if True.
        """

        raise NotImplementedError



class Condistribution(_BaseDistribution):
    r"""
    Abstract class for probability density functions :math:`p(x|y)=c*\tilde{p}(x|y)`. When defining a distribution using this template, the users must implement the following methods:

    - ``sample``: draw samples from the PDF.
    - ``evaluate_density``: evaluate the density function at given points.

    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, num_samples, y: torch.Tensor) -> torch.Tensor:
        r"""
        Draw samples from the distribution :math: `p(\cdot | y)`. The returned samples should be of shape (num_samples, y.shape[0], ...)

        Args:
            num_samples (int): the number of samples to be drawn.
            y (torch.Tensor): the parameters being conditioned on.
        """

        raise NotImplementedError

    @abstractmethod
    def evaluate_density(self, x: torch.Tensor, y: torch.Tensor, in_log: bool) -> torch.Tensor:
        r"""
        Evaluate the density function :math:`\tilde{p}(x|y)` at given :math:`x`. When in_log is True, the logarithm of the density function :math:`log\tilde{p}(x|y)` should be returned. The returned values should be of shape (x.shape[0], y.shape[0]).

        Args:
            x (torch.Tensor): the point(s) at which to evaluate the potential function.
            y (torch.Tensor): the parameters being conditioned on.
            in_log (bool): the returned values are in natural logarithm scale if True.
        """

        raise NotImplementedError
