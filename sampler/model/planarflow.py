from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from .._common import BiProbTrans, Distribution, Func
import warnings

class PlanarFlow(BiProbTrans):
    r"""
    The Planar flow model described in ..[Rezende2015].

    """
    def __init__(self, dim: int,
                 num_trans: int,
                 alpha_lr: Optional[float] = 0.005,
                 alpha_iter: Optional[int] = 8000,
                 alpha_threshold: Optional[float] = 1e-7,
                 p_base: Optional[Distribution] = None):
        super().__init__(p_base=p_base)
        self.dim = dim
        self.num_trans = num_trans
        self.w = nn.Parameter(torch.randn(num_trans, dim))
        self.b = nn.Parameter(torch.randn(num_trans, 1))
        self.u = nn.Parameter(torch.randn(num_trans, dim))

        self._alpha_lr = alpha_lr
        self._alpha_iter = alpha_iter
        self._alpha_threshold = alpha_threshold
        self._reparametrize_u()

    def _reparametrize_u(self):
        prod = torch.sum(self.w * self.u, dim = 1, keepdim = True)
        self.u_hat = self.u + ((-1 + nn.functional.softplus(prod)) - prod) * self.w / torch.sum(self.w ** 2, dim = 1, keepdim=True)


    def _solve_alpha(self, z: torch.Tensor, w: torch.Tensor, u_hat: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        alpha = nn.Parameter(torch.randn(z.shape[0],))
        optimizer = torch.optim.Adam([alpha], lr=self._alpha_lr)
        for iter in range(self._alpha_iter):
            optimizer.zero_grad()
            loss = torch.matmul(z, w) - alpha - torch.matmul(w, u_hat) * torch.tanh(alpha + b)
            loss = torch.mean(loss ** 2)
            loss.backward()
            optimizer.step()
            if loss < self._alpha_threshold:
                break
            elif iter == self._alpha_iter - 1:
                warnings.warn(f"Solving alpha w/ lr = {self._alpha_lr} in PlanarFlow is not converged in "
                              f"{self._alpha_iter} iters. Please increase the number of iters.")

        return alpha.data

    def forward(self, x: torch.Tensor, log_prob: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[
        torch.Tensor, torch.Tensor]:
        for i in range(self.num_trans):
            w, b, u, u_hat = self.w[i], self.b[i], self.u[i], self.u_hat[i]
            neuron = torch.matmul(x, w) + b
            x = x + u_hat.unsqueeze(0) * torch.tanh(neuron.unsqueeze(-1))

            # calculate d tanh(a) / da = 1 / cosh(a) ** 2
            dh = 1 / torch.cosh(neuron) ** 2
            log_prob = log_prob - torch.log(torch.abs(1 + dh * torch.matmul(u_hat, w)))

        return x, log_prob

    def backward(self, z: torch.Tensor, log_prob: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[
        torch.Tensor, torch.Tensor]:
        for i in range(self.num_trans - 1, -1, -1):
            w, b, u, u_hat = self.w[i], self.b[i], self.u[i], self.u_hat[i]
            alpha = self._solve_alpha(z.detach(), w.detach(), u_hat.detach(), b.detach())
            z_parrallel = alpha.unsqueeze(-1) * (w / torch.sum(w ** 2)).unsqueeze(0)
            z_perpendicular = z - z_parrallel - u_hat.unsqueeze(0) * torch.tanh(torch.matmul(z_parrallel, w) + b).unsqueeze(-1)
            z = z_perpendicular + z_parrallel

            # calculate log determinant
            neuron = torch.matmul(z_parrallel, w) + b
            dh = 1 / torch.cosh(neuron) ** 2
            log_prob = log_prob + torch.log(torch.abs(1 + dh * torch.matmul(u_hat, w)))
        return z, log_prob

