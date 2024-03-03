from typing import TypeVar, Callable, Union, Optional
import torch
from abc import ABC, abstractmethod
import math

__all__ = ['Func', 'Distribution']

Func = TypeVar('Func', bound=Callable[[Union[torch.Tensor]], Union[torch.Tensor]])


class Distribution(ABC):
    r"""
    Abstract class for probability density functions :math:`p(x)=c*\hat{p}(x)`. When defining a distribution using this template, the users must implement the following methods:

    - ``__init__``: the initialization method, especially setting the normalization constant :math:`c`. If the normalization constant is not known, it should be set to None. This is also the default behavior.
    - ``sample``: draw samples from the PDF.
    - ``evaluate_density``: evaluate the density function :math:`g(x)` at given points.

    """

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
    def sample(self, num_samples: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
        Draw samples from the distribution :math: `p(\cdot | y)`. The samples should be of shape (num_samples, ...).

        Args:
            num_samples (int): the number of samples to be drawn.
            y (Optional[torch.Tensor]): the parameters being conditioned on. It will only be used to represent drawing samples from a conditional distribution.
        """
        pass

    @abstractmethod
    def evaluate_density(self, x: Union[float, torch.Tensor],
                         y: Optional[torch.Tensor] = None,
                         in_log: Optional[bool] = False) -> torch.Tensor:
        r"""
        Evaluate the density function :math:`\hat{p}(x|y)` at given :math:`x`. When in_log is True, the logarithm of the density function :math:`log\hat{p}(x|y)` should be returned.

        Args:
            x (Union[float, torch.Tensor]): the point(s) at which to evaluate the potential function.
            y (Optional[torch.Tensor]): the parameters being conditioned on. It will be used when evaluating the density of a conditional distribution.
            in_log (bool): the returned values are in natural logarithm scale if True.
        """
        pass

    def __call__(self, x: torch.Tensor,
                 y: Optional[torch.Tensor] = None,
                 in_log: Optional[bool] = False) -> torch.Tensor:
        r"""
        Evaluate the probability distribution :math: `p(x|y)=c*\hat{p}(x|y)` at given :math: `x`. When the normalization constant is known, :math:`p(x|y)/logp(x|y)` will be returned if in_log is False / True. Alternatively, when the normalization constant is None, :math:`\hat{p}(x|y) / log(\hat{p}(x|y))` will be returned if in_log is False / True.

        Args:
            x (Union[float, torch.Tensor]): the point(s) at which to evaluate the PDF.
            y (Optional[torch.Tensor]): the parameters being conditioned on. It will be used when evaluating the density of a conditional distribution.
            in_log (bool): whether to return the logarithm of the PDF.
        """

        result = self.evaluate_density(x, y, in_log)
        if self.norm is not None:
            result = result * self.norm if not in_log else result + math.log(self.norm)
        return result
