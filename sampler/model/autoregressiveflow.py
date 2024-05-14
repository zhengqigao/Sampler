import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
from .._common import BiProbTrans, Distribution

class MAF(BiProbTrans):
    r"""
    Masked Autoregressive Flow described in ..[Papamakarios2017].
    """
    def __init__(self, dim: int,
                 autoregressive_net: Optional[nn.Module] = None,
                 p_base: Optional[Distribution] = None):
        super().__init__()
        self.dim = dim
        self.autoregressive_net = autoregressive_net
        self.p_base = p_base

    def forward(self, x: torch.Tensor,
                log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, alpha = self.autoregressive_net(x)
        z = x * torch.exp(alpha) + mu
        log_det = log_det - torch.sum(alpha, dim=1)
        return z, log_det


    def backward(self, z: torch.Tensor,
                 log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
