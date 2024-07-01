import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
from .._common import BiProbTrans, Distribution
import math
from sampler.model.affinecouplingflow import AffineCouplingFlow

class Actnorm(BiProbTrans):
    r"""
    Actnorm described in ..[kingma2018glow].
    """

    def __init__(self, num_features: int,
                 p_base: Optional[Distribution] = None):
        r"""

        Args:
            num_features (int): The number of features， C from an expected input of size (N, C, ...), e.g., (N, C, H, W) or (N, C,).
            p_base (Optional[Distribution], optional): The base distribution. Defaults to None.
        """

        super().__init__()
        self.num_features = num_features
        self.p_base = p_base

        self.scale = nn.Parameter(torch.empty(num_features))
        self.shift = nn.Parameter(torch.empty(num_features))

        self.reset_parameters()

    def reset_parameters(self, batch_data: Optional[torch.Tensor] = None) -> None:
        if batch_data is None:
            nn.init.ones_(self.scale)
            nn.init.zeros_(self.shift)
        else:
            with torch.no_grad():
                x = batch_data.permute(1, 0, *([2] * (batch_data.dim() - 2))).reshape(self.num_features, -1)
                mean = x.mean(dim=1)
                std = x.std(dim=1)
                nn.init.constant_(self.shift, -mean)
                nn.init.constant_(self.scale, 1 / std)

    def forward(self, x: torch.Tensor,
                log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        n, c, *remain = x.shape

        shift = self.shift.view(1, c, *([1] * len(remain)))
        scale = self.scale.view(1, c, *([1] * len(remain)))

        z = x * scale + shift
        log_det = log_det + torch.sum(torch.log(torch.abs(self.scale))) * (1 if len(remain) == 0 else math.prod(remain))
        return z, log_det

    def backward(self, z: torch.Tensor,
                 log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        n, c, *remain = z.shape

        shift = self.shift.view(1, c, *([1] * len(remain)))
        scale = self.scale.view(1, c, *([1] * len(remain)))

        x = (z - shift) / scale
        log_det = log_det - torch.sum(torch.log(torch.abs(self.scale))) * (1 if len(remain) == 0 else math.prod(remain))
        return x, log_det


class Inv1by1Conv(nn.Module):
    r"""
    The Invertible 1x1 Convolution described in ..[kingma2018glow].
    """

    def __init__(self, num_features: int,
                 bias: bool = False,
                 p_base: Optional[Distribution] = None):
        r"""

        Args:
            num_features (int): The number of features， C from an expected input of size (N, C, ...), e.g., (N, C, H, W) or (N, C,).
            bias (bool, optional): Whether to use bias. Defaults to False.
            p_base (Optional[Distribution], optional): The base distribution. Defaults to None.
        """

        super().__init__()
        self.num_features = num_features
        self.p_base = p_base
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(num_features, num_features))
        self.bias = nn.Parameter(torch.empty(num_features)) if bias else None
        self.register_buffer('permutation', torch.eye(num_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""
        Initialize the weight and bias. Note that the original implementation requires to use the LU factorization,
        """
        nn.init.orthogonal_(self.weight)
        LU, pivots = torch.linalg.lu_factor(self.weight)
        P, L, U = torch.lu_unpack(LU, pivots)

        self.weight = nn.Parameter(torch.tril(L, -1) + torch.triu(U, 0))
        self.permutation = P

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def reparametrize_u(self, weight: torch.Tensor, permutation: torch.Tensor, inverse: bool) -> torch.Tensor:
        l, u = torch.tril(weight, -1) + torch.eye(*weight.shape), torch.triu(weight)
        if not inverse:
            return torch.matmul(permutation, torch.matmul(l, u))
        else:
            p_inv = permutation.t()
            l_inv = torch.linalg.solve_triangular(l, torch.eye(*l.shape), upper=False)
            u_inv = torch.linalg.solve_triangular(u, torch.eye(*u.shape), upper=True)
            return torch.matmul(u_inv, torch.matmul(l_inv, p_inv))

    def forward(self, x: torch.Tensor, log_det: torch.Tensor = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        n, c, *remain = x.shape

        weight = self.reparametrize_u(self.weight, self.permutation, inverse=False)
        bias = self.bias.unsqueeze(0).view(1, c, *[1] * len(remain)) if self.bias is not None else 0.0

        z = torch.einsum('nc...,cd->nd...', x, weight) + bias

        log_det = log_det + torch.log(torch.diag(self.weight).abs()).sum() * math.prod(remain)
        return z, log_det

    def backward(self, z: torch.Tensor, log_det: torch.Tensor = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        n, c, *remain = z.shape

        weight = self.reparametrize_u(self.weight, self.permutation, inverse=True)
        bias = self.bias.unsqueeze(0).view(1, c, *[1] * len(remain)) if self.bias is not None else 0.0

        x = torch.einsum('nc...,cd->nd...', z - bias, weight)

        log_det = log_det - torch.log(torch.diag(self.weight).abs()).sum() * math.prod(remain)
        return x, log_det

"""
class glow(BiProbTrans):

    def __init__(self, num_features: int,
                 num_trans: int,
                 dim: int,
                 scale_net: Optional[Union[nn.Module, nn.ModuleList, List, Tuple]] = None,
                 shift_net: Optional[Union[nn.Module, nn.ModuleList, List, Tuple]] = None,
                 keep_dim: Optional[Union[torch.Tensor, List[List[int]]]] = None,
                 p_base: Optional[Distribution] = None):
        super().__init__()

        self.num_trans = num_trans
        self.dim = dim
        self.scale_net = scale_net
        self.shift_net = shift_net
        self.p_base = p_base

        # by default, keep_dim alternates between even and odd indices
        if keep_dim is None:
            self.keep_dim = [list(range(i % 2 == 0, dim, 2)) for i in range(num_trans)]
        elif len(keep_dim) != num_trans:
            raise ValueError(f"keep_dim should have length {self.num_trans}, but got {len(keep_dim)}.")
        else:
            self.keep_dim = keep_dim

        self.transforms = nn.ModuleList([AffineCouplingFlow(dim=self.dim,
                                                            keep_dim=self.keep_dim[i],
                                                            scale_net=self.scale_net[i] if isinstance(self.scale_net,
                                                                                                      (nn.ModuleList,
                                                                                                       List,
                                                                                                       Tuple)) and i < len(
                                                                self.scale_net) else self.scale_net,
                                                            shift_net=self.shift_net[i] if isinstance(self.shift_net,
                                                                                                      (nn.ModuleList,
                                                                                                       List,
                                                                                                       Tuple)) and i < len(
                                                                self.shift_net) else None) for i in range(num_trans)])


    def forward(self):


    def backforward(self):

"""




