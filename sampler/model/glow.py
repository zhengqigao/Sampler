import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
from .._common import BiProbTrans, Distribution
import math


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

        self.weight = torch.tril(L, -1) + torch.triu(U, 0)
        self.permutation = P

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    ## TODO: definitely rewrite this function, it is wrong.
    def reparametrize_u(self, weight: torch.Tensor, permutation: torch.Tensor, inverse: bool) -> torch.Tensor:
        if not inverse:
            l, u = torch.tril(weight,-1) + torch.diag(torch.ones(*weight.shape)), torch.triu(weight)
            return torch.matmul(permutation, torch.matmul(l, u))
        else:
            l, u = torch.tril(weight,-1) + torch.diag(torch.ones(*weight.shape)), torch.triu(weight)
            return torch.matmul(torch.inverse(permutation), torch.matmul(u, l))

    def forward(self, x: torch.Tensor, log_det: torch.Tensor = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        n, c, *remain = x.shape

        weight = self.reparametrize_u(self.weight, self.permutation, inverse = False).unsqueeze(0)
        bias = self.bias.unsqueeze(0) if self.bias is not None else 0.0
        z = torch.matmul(x.view(n, c, -1), weight) + bias
        z = z.view(n, c, *remain)

        log_det = log_det + torch.slogdet(weight.squeeze())[1] * math.prod(remain)
        return z, log_det

    def backward(self, z: torch.Tensor, log_det: torch.Tensor = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        n, c, *remain = z.shape

        weight = self.reparametrize_u(self.weight, self.permutation, inverse = True).unsqueeze(0)
        bias = self.bias.unsqueeze(0)

        x = torch.matmul(z.view(n, c, -1) - bias, weight)
        x = x.view(n, c, *remain)

        log_det = log_det - torch.slogdet(weight.squeeze())[1] * math.prod(remain)
        return x, log_det
