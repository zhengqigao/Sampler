import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
from .._common import BiProbTrans, Distribution


class AffineCouplingFlow(BiProbTrans):
    r"""
    The Affine coupling flow. It keeps the dimensions unchanged specified by the argument `keep_dim`, and the other
    dimensions are transformed by the scale and shift networks. See ..[Dinh2017] for more details.

    Example:
        .. code-block:: python

            >>> couplingflow = AffineCouplingFlow(dim=2, keep_dim=[0], scale_net=nn.Linear(1, 1), shift_net=nn.Linear(1, 1))
            >>> x = torch.rand(10, 2)
            >>> x_, diff_log_det = couplingflow.backward(*couplingflow.forward(x, 0))
            >>> diff = x - x_
            >>> print(f"diff = {torch.max(torch.abs(diff))}, diff_log_det = {torch.max(torch.abs(diff_log_det))}")
    """

    def __init__(self, dim: int,
                 keep_dim: Union[List[int], Tuple[int], torch.Tensor],
                 scale_net: Optional[nn.Module] = None,
                 shift_net: Optional[nn.Module] = None,
                 p_base: Optional[Distribution] = None):
        super().__init__()

        self.dim = dim

        keep_dim, all_dim = torch.tensor(keep_dim, dtype=torch.int), torch.arange(dim)
        if ~torch.all(torch.isin(keep_dim, all_dim)):
            raise ValueError(f"keep_dim should be a subset of [0,{dim-1}), but got {keep_dim}.")

        self.register_buffer("keep_dim", keep_dim)
        self.register_buffer("trans_dim", all_dim[~torch.isin(all_dim, keep_dim)])

        self.scale_net = scale_net
        self.shift_net = shift_net

        self.p_base = p_base

    def forward(self, x: torch.Tensor,
                log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        x_keep, x_trans = x[:, self.keep_dim], x[:, self.trans_dim]
        s = self.scale_net(x_keep) if self.scale_net is not None else torch.zeros_like(x_trans)
        t = self.shift_net(x_keep) if self.shift_net is not None else torch.zeros_like(x_trans)

        z = torch.zeros_like(x)
        z[:, self.keep_dim] = x_keep
        z[:, self.trans_dim] = torch.exp(s) * x_trans + t
        log_det = log_det - torch.log(torch.prod(torch.exp(s), dim=1))

        return z, log_det

    def backward(self, z: torch.Tensor,
                 log_det: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        z_keep, z_trans = z[:, self.keep_dim], z[:, self.trans_dim]
        s = self.scale_net(z_keep) if self.scale_net is not None else torch.zeros_like(z_trans)
        t = self.shift_net(z_keep) if self.shift_net is not None else torch.zeros_like(z_trans)

        x = torch.zeros_like(z)
        x[:, self.keep_dim] = z_keep
        x[:, self.trans_dim] = (z_trans - t) * torch.exp(-s)
        log_det = log_det + torch.log(torch.prod(torch.exp(s), dim=1))
        return x, log_det  # Our implementation guarantees: x, a = model.backward(*model.forward(x, log_det = a))


class RealNVP(BiProbTrans):
    r"""
    The RealNVP model. It is a sequence of `num_trans` affine coupling flows. By default, we change the dimensions with
    the odd indices in the first transformation, and the even indices in the second transformation, and keep this
    alternating pattern. It also accepts customized `keep_dim` for each transformation. The `scale_net` and `shift_net`
    are both a list of modules of the same length as `num_trans` to allow for customized architectures. See ..[Dinh2017]
    for more details.

    Example:
        .. code-block:: python

            >>> realnvp = RealNVP(num_trans = 3, dim=2, scale_net=nn.Linear(1, 1), shift_net=nn.Linear(1, 1))
            >>> x = torch.rand(10, 2)
            >>> x_, diff_log_det = realnvp.backward(*realnvp.forward(x, 0))
            >>> diff = x - x_
            >>> print(f"diff = {torch.max(torch.abs(diff))}, diff_log_det = {torch.max(torch.abs(diff_log_det))}")
    """

    def __init__(self, num_trans: int,
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


class NICE(RealNVP):
    r"""
    The NICE model described in ..[dinh2015nice]. It is a special case of RealNVP where the scale networks are none.

    Example:
        .. code-block:: python

            >>> nice = NICE(num_trans = 3, dim=2, shift_net=nn.Linear(1, 1))
            >>> x = torch.rand(10, 2)
            >>> x_, diff_log_det = nice.backward(*nice.forward(x, 0))
            >>> diff = x - x_
            >>> print(f"diff = {torch.max(torch.abs(diff))}, diff_log_det = {torch.max(torch.abs(diff_log_det))}")
    """

    def __init__(self, num_trans: int,
                 dim: int,
                 shift_net: Optional[Union[nn.Module, nn.ModuleList, List, Tuple]] = None,
                 keep_dim: Optional[List[List[int]]] = None,
                 p_base: Optional[Distribution] = None):
        super().__init__(num_trans, dim, None, shift_net, keep_dim, p_base)
