from typing import TypeVar, Callable, Union, Optional
import torch
from abc import ABC, abstractmethod
import math

__all__ = ['_Func', 'Distribution']


_Func = TypeVar('_Func', bound=Callable[[Union[float, torch.Tensor]], Union[float, torch.Tensor]])


class Distribution(ABC):
    r"""
    Abstract class for probability density functions :math:`p(x)=c*g(x)`. When defining a distribution using this template, the users must implement the following methods:

    - __init__: the initialization method, especially setting the normalization constant :math:`c`. If the normalization constant is not known, it should be set to None. This is also the default behavior.
    - sample: draw samples from the PDF.
    - evaluate_density: evaluate the density function :math:`g(x)` at given points.

    """

    def __init__(self):
        self._const = None

    @property
    def const(self):
        if not hasattr(self, '_const'):
            raise AttributeError("The normalization constant is not provided, Please use `self._const = None` if unknown.")
        return self._const

    @const.setter
    def const(self, value):
        if value is not None and value <= 0:
            raise ValueError("The normalization constant must be positive.")
        else:
            self._const = value

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        r"""
        Draw samples from the PDF. The samples should be of shape (num_samples, ...).

        Args:
            num_samples (int): the number of samples to be drawn.

        """
        pass

    @abstractmethod
    def evaluate_density(self, x: Union[float, torch.Tensor], in_log: Optional[bool] = False) -> Union[float, torch.Tensor]:
        r"""
        Evaluate the density function :math:`g(x)` at given points. When in_log is True, the logarithm of the density function :math:`log(g(x))` should be returned.

        Args:
            x (Union[float, torch.Tensor]): the point(s) at which to evaluate the potential function.
            in_log (bool): the returned values are in natural logarithm scale if True.
        """
        pass


    def __call__(self, x: Union[float, torch.Tensor],
                      in_log: Optional[bool] = False) -> Union[float, torch.Tensor]:
        r"""
        Evaluate the probability distribution :math: `p(x)=c*g(x)` at given points. When the normalization constant is known, :math:`p(x)/logp(x)` will be returned if in_log is False / True. Alternatively, when the normalization constant is None, :math:`g(x) / log(g(x))` will be returned if in_log is False / True.

        Args:
            x (Union[float, torch.Tensor]): the point(s) at which to evaluate the PDF.
            in_log (bool): whether to return the logarithm of the PDF.
        """

        result = self.evaluate_density(x, in_log)
        if self.const is not None:
            result = result * self.const if not in_log else result + math.log(self.const)
        return result
