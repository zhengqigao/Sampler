from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from .._utils import _ModuleWrapper
from .._common import UniProbTrans, BiProbTrans, Distribution, Func


class PlanarFlow(UniProbTrans):
    def __init__(self, dim: int,
                 num_trans: int,
                 activation: Optional[Union[nn.Module, Func]] = nn.ReLU(),
                 p_base: Optional[Distribution] = None):
        super(PlanarFlow, self).__init__(p_base=p_base)
        self.dim = dim
        self.num_trans = num_trans
        self.transforms = nn.ModuleList([nn.Sequential(nn.Linear(dim, 1),
                                                       activation if isinstance(activation, nn.Module) else _ModuleWrapper(activation),
                                                       nn.Linear(1, dim, bias=False)) for _ in range(num_trans)])

    def forward(self, x: torch.Tensor, log_prob: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[
        torch.Tensor, torch.Tensor]:
        for i in range(len(self.transforms)):
            x_new = self.transforms[i](x)
            log_prob += 1.0 + torch.autograd.grad(x_new.sum(), x)[0]
            x = x + x_new
        return x, log_prob

