from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from .._utils import _ModuleWrapper
from .._common import UniProbTrans, BiProbTrans, Distribution, Func


class PlanarFlow(BiProbTrans):
    r"""
    The Planar flow model described in ..[Rezende2015].

    """
    def __init__(self, dim: int,
                 num_trans: int,
                 alpha_lr: Optional[float] = 0.01,
                 alpha_iter: Optional[int] = 10,
                 p_base: Optional[Distribution] = None):
        super().__init__(p_base=p_base)
        self.dim = dim
        self.num_trans = num_trans
        self.w = nn.Parameter(torch.randn(num_trans, dim))
        self.b = nn.Parameter(torch.randn(num_trans, 1))
        self.u = nn.Parameter(torch.randn(num_trans, dim))

        self._alpha_lr = alpha_lr
        self._alpha_iter = alpha_iter
        self._reparametrize_u()

    def _reparametrize_u(self):
        prod = torch.sum(self.w * self.u, dim = 1, keepdim = True)
        self.u_hat = self.u + ((-1 + nn.functional.softplus(prod)) - prod) * self.w / torch.sum(self.w ** 2, dim = 1)

    def _solve_alpha(self, z: torch.Tensor, w: torch.Tensor, u_hat: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        alpha = nn.Parameter(1)
        loss = torch.matmul(z, w) - alpha - torch.matmul(w, u_hat) * torch.tanh(alpha + b)
        optimizer = torch.optim.Adam([alpha], lr=self._alpha_lr)
        for _ in range(self._alpha_iter):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return alpha

    def forward(self, x: torch.Tensor, log_prob: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[
        torch.Tensor, torch.Tensor]:
        for i in range(len(self.transforms)):
            w, b, u, u_hat = self.w[i], self.b[i], self.u[i], self.u_hat[i]
            neuron = torch.matmul(x, w) + b
            x += u_hat[i] * torch.tanh(neuron)

            # calculate d tanh(a) / da = 1 / cosh(a) ** 2
            dh = 1 / torch.cosh(neuron) ** 2
            log_prob -= torch.log(torch.abs(1 + dh * torch.matmul(u_hat, w)))

        return x, log_prob

    def backward(self, z: torch.Tensor, log_prob: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[
        torch.Tensor, torch.Tensor]:
        for i in range(len(self.transforms) - 1, -1, -1):
            w, b, u, u_hat = self.w[i], self.b[i], self.u[i], self.u_hat[i]
            alpha = self._solve_alpha(z.detach(), w.detach(), u_hat.detach(), b.detach())
            z_parrallel = alpha * w / torch.sum(w ** 2)
            z_perpendicular = z - z_parrallel - u_hat * torch.tanh(torch.matmul(z_parrallel, w) + b)
            z = z_perpendicular + z_parrallel

            # calculate log determinant
            neuron = torch.matmul(z_parrallel, w) + b
            dh = 1 / torch.cosh(neuron) ** 2
            log_prob += torch.log(torch.abs(1 + dh * torch.matmul(u_hat, w)))
        return z, log_prob

