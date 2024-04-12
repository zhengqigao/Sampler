import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional


class FlowTransform(nn.Module):
    r"""
    The probabilistic transformation building block used in flow models. It has a forward and a backward method.
    The forward method transforms the input tensor x to z, and outputs the log probability of the transformation.
    The backward method transforms z back to x, and outputs the log probability.

    .. Example::
        >>> flowtransform = FlowTransform(dim=2, keep_dim=[0], scale_net=nn.Linear(1, 1), shift_net=nn.Linear(1, 1))
        >>> x = torch.rand(10, 2)
        >>> x_, diff_log_det = flowtransform.backward(*flowtransform.forward(x, 0))
        >>> diff = x - x_
        >>> print(f"diff = {torch.max(torch.abs(diff))}, diff_log_det = {torch.max(torch.abs(diff_log_det))}")
    """

    def __init__(self, dim: int, keep_dim: List[int], scale_net: Optional[nn.Module] = None,
                 shift_net: Optional[nn.Module] = None):
        super(FlowTransform, self).__init__()

        if not set(keep_dim).issubset(set(range(dim))):
            raise ValueError(f"keep_dim should be a subset of [0, {dim}), but got {keep_dim}.")

        self.dim = dim
        self.keep_dim = keep_dim
        self.trans_dim = sorted(list(set(range(dim)) - set(keep_dim)))

        self.scale_net = scale_net
        self.shift_net = shift_net

    def forward(self, x, log_det = 0):
        x_keep, x_trans = x[:, self.keep_dim], x[:, self.trans_dim]
        s = self.scale_net(x_keep) if self.scale_net is not None else torch.zeros_like(x_trans)
        t = self.shift_net(x_keep) if self.shift_net is not None else torch.zeros_like(x_trans)

        z = torch.zeros_like(x)
        z[:, self.keep_dim] = x_keep
        z[:, self.trans_dim] = torch.exp(s) * x_trans + t
        log_det += torch.log(torch.prod(torch.exp(s), dim=1))

        return z, log_det

    def backward(self, z, log_det = 0):
        z_keep, z_trans = z[:, self.keep_dim], z[:, self.trans_dim]
        s = self.scale_net(z_keep) if self.scale_net is not None else torch.zeros_like(z_trans)
        t = self.shift_net(z_keep) if self.shift_net is not None else torch.zeros_like(z_trans)

        x = torch.zeros_like(z)
        x[:, self.keep_dim] = z_keep
        x[:, self.trans_dim] = (z_trans - t) * torch.exp(-s)
        log_det -= torch.log(torch.prod(torch.exp(s), dim=1))
        return x, log_det  # Our implementation guarantees: x, a = model.backward(*model.forward(x, log_det = a))


class BaseNormalizingFlow(nn.Module):
    def __init__(self, num_trans: int,
                 dim: int,
                 scale_net: nn.ModuleList,
                 shift_net: nn.ModuleList,
                 keep_dim: Optional[List[List[int]]] = None):
        super(BaseNormalizingFlow, self).__init__()

        self.num_trans = num_trans
        self.dim = dim
        self.scale_net = scale_net
        self.shift_net = shift_net

        # by default, keep_dim alternates between even and odd indices
        if keep_dim is None:
            self.keep_dim = [list(range(i % 2 == 0, dim, 2)) for i in range(num_trans)]
        elif len(keep_dim) != num_trans: ## TODO: might need to add more checks, such as every element in keep_dim should be a subset of [0, dim)
            raise ValueError(f"keep_dim should have length {self.num_trans}, but got {len(keep_dim)}.")
        else:
            self.keep_dim = keep_dim

        self.transforms = nn.ModuleList([FlowTransform(dim=self.dim,
                                                       keep_dim=self.keep_dim[i],
                                                       scale_net=self.scale_net[i] if self.scale_net is not None and i < len(
                                                           self.scale_net) else None,
                                                       shift_net=self.shift_net[i] if self.shift_net is not None and i < len(
                                                           self.shift_net) else None) for i in range(num_trans)])

    def forward(self, x, log_det = 0):
        for i in range(len(self.transforms)):
            x, ld = self.transforms[i].forward(x)
            log_det += ld
        return x, log_det

    def backward(self, z, log_det = 0):
        for i in range(len(self.transforms)):
            z, ld = self.transforms[-1 - i].backward(z)
            log_det += ld
        return z, log_det  # Our implementation guarantees: x, a = model.backward(*model.forward(x, log_det = a))
