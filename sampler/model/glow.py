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
        self.p_base = p_base  ## what is this for?
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


class Glowblock(BiProbTrans):
    r"""
    Basic step in the Glow flow

    - ActNorm
    - Invertible1x1Conv
    - MaskedAffineFlow
    """

    def __init__(self,
                 num_features: int,
                 num_trans: int,
                 scale_net: Optional[Union[nn.Module, nn.ModuleList, List, Tuple]] = None,
                 shift_net: Optional[Union[nn.Module, nn.ModuleList, List, Tuple]] = None,
                 keep_dim: Optional[Union[torch.Tensor, List[int]]] = None,
                 p_base: Optional[Distribution] = None):
        super().__init__()

        self.num_features = num_features
        self.num_trans = num_trans
        self.scale_net = scale_net
        self.shift_net = shift_net
        self.p_base = p_base

        # by default, keep_dim is the first half vector
        if keep_dim is None:
            self.keep_dim = [i for i in range(num_features // 2)]
        elif len(keep_dim) != num_trans:
            raise ValueError(f"keep_dim should have length {self.num_trans}, but got {len(keep_dim)}.")
        else:
            self.keep_dim = keep_dim

        self.transforms = []
        self.transforms.append(AffineCouplingFlow(dim=self.num_features,
                                                  keep_dim=self.keep_dim,
                                                  scale_net=self.scale_net,
                                                  glow_mode=True))
        self.transforms.append(Inv1by1Conv(self.num_features))
        self.transforms.append(Actnorm(self.num_features))


    def forward(self, x: torch.Tensor,
                log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            x, log_det = transform.forward(x, log_det)
            # print(log_det)
        return x, log_det

    def backward(self, z: torch.Tensor,
                 log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in reversed(self.transforms):
            z, log_det = transform.backward(z, log_det)
        return z, log_det


# TODO: find a way to easily understand this operation
class Squeeze(nn.Module):
    r"""
    Squeeze operation, Reference to vince
    """

    def __init__(self):
        """
        from normalizing flow pkd
        """
        super().__init__()

    def backward(self, z: torch.Tensor, log_det: torch.Tensor = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        s = z.size()
        z = z.view(*s[:2], s[2] // 2, 2, s[3] // 2, 2)
        z = z.permute(0, 1, 3, 5, 2, 4).contiguous()
        z = z.view(s[0], 4 * s[1], s[2] // 2, s[3] // 2)
        return z, log_det

    def forward(self, z: torch.Tensor, log_det: torch.Tensor = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        s = z.size()
        z = z.view(s[0], s[1] // 4, 2, 2, s[2], s[3])  # check if our views is the sam
        z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
        z = z.view(s[0], s[1] // 4, 2 * s[2], 2 * s[3])
        return z, log_det


class Split(BiProbTrans):

    def __init__(self, mode="pos"):
        super().__init__()
        self.mode = mode

    def backward(self, x: torch.Tensor,
                log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "pos":
            x1, x2 = x.chunk(2, dim=1)
        elif self.mode == "inv":
            x2, x1 = x.chunk(2, dim=1)
        return [x1, x2], log_det

    def forward(self, z: torch.Tensor,
                 log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        z1, z2 = z
        if self.mode == "pos":
            z = torch.cat([z1, z2], 1)
        elif self.mode == "inv":
            z = torch.cat([z2, z1], 1)
        return z, log_det


class MultiscaleFlow(BiProbTrans):
    """
    Normalizing Flow model with multiscale architecture, see RealNVP or Glow paper
    """

    def __init__(self, p_base, flows, splits, transform=None, class_cond=True):
        super().__init__()
        self.p_base = nn.ModuleList(p_base)
        self.num_levels = len(self.p_base)
        self.flows = torch.nn.ModuleList([nn.ModuleList(flow) for flow in flows])
        self.splits = torch.nn.ModuleList(splits)
        self.transform = transform
        self.class_cond = class_cond

    def backward(self, x: torch.Tensor,
                log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # log_det = torch.zeros(len(x), dtype=x.dtype, device=x.device) ## why it is a list?

        z = [None] * len(self.p_base)
        for i in range(len(self.p_base)):
            for flow in self.flows[i]:
                x, log_det = flow.backward(x, log_det)
            if i == len(self.p_base) - 1:
                z[i] = x
            else:
                [x, z[i]], log_det = self.splits[i].backward(x, log_det)
        return z, log_det

    def forward(self, z: list[torch.Tensor],
                 log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:

        for i in range(len(self.p_base)-1, -1, -1):
            if i == len(self.p_base)-1:
                z_ = z[i]
            else:
                z_, log_det = self.splits[i]([z_, z[i]], log_det)
                #print(z_)
            for flow in reversed(self.flows[i]): #TODO: reversed should be revised
                z_, log_det = flow(z_, log_det)
        #TODO: transform
        #if self.transform is not None:
        #    z_, log_det_ = self.transform(z_)
        #   log_det += log_det_
        return z_, log_det

    def log_prob(self, z, y):
        #TODO: merge in our flow
        x, log_det = self.backward(z)
        print(log_det)
        for i in range(len(self.p_base)):
            if self.class_cond:
                log_det = log_det + self.p_base[i].log_prob(x[i], y)
            else:
                log_det += self.p_base[i].log_prob(x[i])
        return log_det
    def sample(self, num_samples=1, y=None, temperature=None):
        if temperature is not None:
            self.set_temperature(temperature)
        log_q = 0
        for i in range(len(self.p_base)-1, -1, -1):
            if self.class_cond:
                z_, log_q = self.p_base[i](num_samples, y, log_q)
            else:
                z_, log_q = self.p_base[i](num_samples, log_q)
            if i == len(self.p_base)-1:
                z = z_
            else:
                z, log_q = self.splits[i]([z, z_], log_q)
            #print(z)
            for flow in reversed(self.flows[i]):
                z, log_q = flow(z, log_q)
        return z, log_q
