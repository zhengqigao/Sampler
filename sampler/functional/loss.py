import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
from .._common import BiProbTrans, Distribution, Func, _bpt_decorator
from ..base import score_estimator


class KLGenLoss(nn.Module):
    r"""
    The KL divergence loss KL[p||q] used in a generative task, where q is the learnable distribution and p is the target
    distribution. It can be reduced to negative log likelihood :math: `1/N \sum log q(z_n)`, where :math: `z_n` is the n-th
    sample in the training dataset.

    """

    def __init__(self, reduction: Optional[str] = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, model: BiProbTrans, z: torch.Tensor) -> torch.Tensor:
        x, log_q = model.log_prob(z)
        if self.reduction == 'mean':
            result = -torch.mean(log_q)
        elif self.reduction == 'sum':
            result = -torch.sum(log_q)
        elif self.reduction == 'none':
            result = -log_q
        else:
            raise ValueError(f"reduction should be 'mean', 'sum' or 'none', but got {self.reduction}.")
        return result


class KLDenLoss(nn.Module):
    r"""
    The KL divergence loss KL[q||p] used in a density estimation task, where q is learnable and p is the target.

    """

    def __init__(self, log_p: Func, reduction: Optional[str] = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.log_p = log_p

    def forward(self, model: BiProbTrans, num_samples: int) -> torch.Tensor:
        sample, log_q = model.sample(num_samples)
        if self.reduction == 'mean':
            result = torch.mean(log_q - self.log_p(sample))
        elif self.reduction == 'sum':
            result = torch.sum(log_q - self.log_p(sample))
        elif self.reduction == 'none':
            result = log_q - self.log_p(sample)
        else:
            raise ValueError(f"reduction should be 'mean', 'sum' or 'none', but got {self.reduction}.")
        return result


class ScoreDenLoss(nn.Module):
    def __init__(self, log_p: Func, reduction: Optional[str] = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.log_p = log_p

    @_bpt_decorator
    def forward(self, model: Union[Distribution, BiProbTrans], num_samples: int) -> torch.Tensor:
        return score_estimator(num_samples, model, lambda x: model(x) - self.log_p(x), self.reduction)
