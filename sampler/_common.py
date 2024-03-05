from typing import TypeVar, Callable, Union, Optional
import torch
from abc import ABC, abstractmethod
import math
from torch.distributions import Distribution as TorchDistribution

__all__ = ['Func', 'Distribution', 'Condistribution', 'Wrapper']

Func = TypeVar('Func', bound=Callable[[Union[torch.Tensor]], Union[torch.Tensor]])


class _BaseDistribution(ABC):

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
        if value is not None and value <= 0:
            raise ValueError("The normalization constant must be positive.")
        else:
            self._norm = value

    @abstractmethod
    def sample(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def evaluate_density(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        r"""
        Evaluate the probability distribution :math: `p(x|y)=c*\tilde{p}(x|y)` at given :math: `x`. When the normalization constant is known, :math:`p(x|y)/logp(x|y)` will be returned if in_log is False / True. Alternatively, when the normalization constant is None, :math:`\tilde{p}(x|y) / log(\tilde{p}(x|y))` will be returned if in_log is False / True.

        Args:
            x (Union[float, torch.Tensor]): the point(s) at which to evaluate the PDF.
            y (Optional[torch.Tensor]): the parameters being conditioned on. It will be used when evaluating the density of a conditional distribution.
            in_log (bool): whether to return the logarithm of the PDF.
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
        Evaluate the density function :math:`\tilde{p}(x)` at given :math:`x`. When in_log is True, the logarithm of the density function :math:`log\tilde{p}(x)` should be returned.

        Args:
            x (torch.Tensor): the point(s) at which to evaluate the potential function.
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
        Draw samples from the distribution :math: `p(\cdot | y)`. The samples should be of shape (num_samples, ...).

        Args:
            num_samples (int): the number of samples to be drawn.
            y (torch.Tensor): the parameters being conditioned on.
        """

        raise NotImplementedError

    @abstractmethod
    def evaluate_density(self, x: torch.Tensor, y: torch.Tensor, in_log: bool) -> torch.Tensor:
        r"""
        Evaluate the density function :math:`\tilde{p}(x|y)` at given :math:`x`. When in_log is True, the logarithm of the density function :math:`log\tilde{p}(x|y)` should be returned.

        Args:
            x (Union[float, torch.Tensor]): the point(s) at which to evaluate the potential function.
            y (torch.Tensor): the parameters being conditioned on.
            in_log (bool): the returned values are in natural logarithm scale if True.
        """

        raise NotImplementedError


class Wrapper(Distribution):
    def __init__(self, distribution: TorchDistribution):
        super().__init__()
        self.distribution = distribution
        self.norm = 1.0
    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    def evaluate_density(self, x: torch.Tensor, in_log: bool = False) -> torch.Tensor:
        if in_log:
            return self.distribution.log_prob(x)
        else:
            return torch.exp(self.distribution.log_prob(x))

