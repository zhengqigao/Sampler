import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
from .._common import BiProbTrans, Distribution, Func

class KLGenLoss(nn.Module):
    def __init__(self, reduction: Optional[str] = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, model: BiProbTrans, z: torch.Tensor) -> torch.Tensor:
        x, log_q = model.log_prob(z)
        if self.reduction == 'mean':
            loss = -torch.mean(log_q)
        elif self.reduction == 'sum':
            loss = -torch.sum(log_q)
        elif self.reduction == 'none':
            loss = -log_q
        else:
            raise ValueError(f"reduction should be 'mean', 'sum' or 'none', but got {self.reduction}.")
        return loss

class KLDenLoss(nn.Module):
    def __init(self, reduction: Optional[str] = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, model: BiProbTrans, num_samples: int, log_p: Func):
        sample, log_prob = model.sample(num_samples)
        loss = torch.mean(log_prob + log_p(sample))
        return loss

