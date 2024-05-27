import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
from .._common import BiProbTrans, Distribution
from .made import MADE


class MAFLayer(BiProbTrans):
    def __init__(self, dim: int,
                 made: MADE,
                 p_base: Optional[Distribution] = None):
        super().__init__()
        self.dim = dim
        self.made = made
        self.p_base = p_base

    def forward(self, x: torch.Tensor,
                log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        param = self.made(x)
        mu, alpha = torch.chunk(param, 2, dim=1)
        z = (x - mu) * torch.exp(-alpha)
        log_det = log_det - torch.sum(alpha, dim=1)
        return z, log_det

    def backward(self, z: torch.Tensor,
                 log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros_like(z)
        for i in range(self.dim):
            param = self.made(x)
            mu, alpha = torch.chunk(param, 2, dim=1)
            x[:, i] = z[:, i] * torch.exp(alpha[:, i]) + mu[:, i]
        log_det = log_det + torch.sum(alpha, dim=1)
        return x, log_det

class MAF(BiProbTrans):
    r"""
    Masked Autoregressive Flow described in ..[Papamakarios2017].
    """

    def __init__(self, dim: int,
                 num_trans: int,
                 made: Optional[Union[nn.Module, nn.ModuleList, List, Tuple]] = None,
                 p_base: Optional[Distribution] = None):
        super().__init__()
        self.dim = dim
        self.made = made
        self.p_base = p_base

        self.transforms = nn.ModuleList([MAFLayer(dim=self.dim,
                                                  made = self.made[i] if isinstance(self.made, (nn.ModuleList, list, tuple)) and i < len(
                                                        self.made) else self.made,) for i in range(num_trans)])

    def forward(self, x: torch.Tensor,
                log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            x, ld = transform.forward(x)
            log_det = log_det + ld
        return x, log_det

    def backward(self, z: torch.Tensor,
                 log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in reversed(self.transforms):
            z, ld = transform.backward(z)
            log_det = log_det + ld
        return z, log_det