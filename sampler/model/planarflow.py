from typing import Optional, Union, Tuple, Any
import torch
import torch.nn as nn
from .._common import BiProbTrans, Distribution, Func
import warnings
import torch.nn.init as init
import math


class PlanarFlow(BiProbTrans):
    r"""
    The Planar flow model described in ..[Rezende2015]. The invertibility conditions described in their Appendix are implemented.

    """

    def __init__(self, dim: int,
                 num_trans: int,
                 alpha_iter: Optional[int] = 10000,
                 alpha_threshold: Optional[float] = 1e-9,
                 p_base: Optional[Distribution] = None):
        super().__init__(p_base=p_base)
        self.dim = dim
        self.num_trans = num_trans
        self.w = nn.Parameter(torch.empty(num_trans, dim))
        self.b = nn.Parameter(torch.empty(num_trans, 1))
        self.u = nn.Parameter(torch.empty(num_trans, dim))
        self.reset_parameters()

        self._alpha_iter = alpha_iter
        self._alpha_threshold = alpha_threshold

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.w, a=math.sqrt(5))
        init.kaiming_uniform_(self.u, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.b, -bound, bound)

        # init.uniform_(self.w, -0.01, 0.01)
        # init.uniform_(self.u, -0.01, 0.01)
        # init.uniform_(self.b, -0.01, 0.01)

    def reparametrize_u(self) -> torch.Tensor:
        prod = torch.sum(self.w * self.u, dim=1, keepdim=True)
        u_hat = self.u + ((-1 + nn.functional.softplus(prod)) - prod) * self.w / torch.sum(self.w ** 2, dim=1,
                                                                                           keepdim=True)
        return u_hat

    def solve_alpha(self, c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return AlphaSolver.apply(c, a, b, self._alpha_iter, self._alpha_threshold)

    def forward(self, x: torch.Tensor, log_prob: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[
        torch.Tensor, torch.Tensor]:
        u_hat = self.reparametrize_u()
        for i in range(self.num_trans):
            w, b, u, u_hat_i = self.w[i], self.b[i], self.u[i], u_hat[i]
            neuron = torch.matmul(x, w) + b
            x = x + u_hat_i.unsqueeze(0) * torch.tanh(neuron.unsqueeze(-1))

            # calculate d tanh(a) / da = 1 / cosh(a) ** 2 = 1 - tanh(a) ** 2
            dh = 1 - torch.tanh(neuron) ** 2
            log_prob = log_prob - torch.log(torch.abs(1 + dh * torch.matmul(u_hat_i, w)))
        return x, log_prob

    def backward(self, z: torch.Tensor, log_prob: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[
        torch.Tensor, torch.Tensor]:
        u_hat = self.reparametrize_u()
        for i in range(self.num_trans - 1, -1, -1):
            w, b, u, u_hat_i = self.w[i], self.b[i], self.u[i], u_hat[i]
            c, a = torch.matmul(z, w), torch.matmul(w, u_hat_i)
            alpha = self.solve_alpha(c, a, b)

            # a simplified equation, instead of calculating parallel and perpendicular components.
            z = z - u_hat_i.unsqueeze(0) * torch.tanh(alpha + b).unsqueeze(-1)
            dh = 1 - torch.tanh(alpha + b) ** 2
            log_prob = log_prob + torch.log(torch.abs(1 + dh * torch.matmul(u_hat_i, w)))
        return z, log_prob


class AlphaSolver(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, c: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                alpha_iter: int,
                alpha_threshold: float, ) -> torch.Tensor:

        # solve alpha defined by the equation: c = alpha + a * tanh(alpha + b).
        # alpha in [c-a, c+a] because tanh is in [-1, 1]
        with torch.no_grad():

            if a.abs() <= 1e-15:
                return c

            alpha = c * torch.ones_like(c)
            for i in range(alpha_iter):
                loss = (c - alpha) / a - torch.tanh(alpha + b)
                dloss = -1 / a - (1 - torch.tanh(alpha + b) ** 2)
                alpha = alpha - loss / dloss
                if (loss ** 2).mean() < alpha_threshold:
                    break
                elif i == alpha_iter - 1:
                    warnings.warn(f"Solving alpha in PlanarFlow does not converge in "
                                  f"{alpha_iter} iters, with final loss = {(loss ** 2).mean().item():.2e}."
                                  f" Please increase the number of iters.")

        ctx.save_for_backward(a, b, c, alpha)
        return alpha

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        a, b, c, alpha = ctx.saved_tensors
        tanh_value = torch.tanh(alpha + b)
        dtanh = 1 - tanh_value ** 2
        dalpha_dc = 1 / (1 + a * dtanh)
        dalpha_db = -a * dtanh / (1 + a * dtanh)
        dalpha_da = -tanh_value / (1 + a * dtanh)

        dl_dc = grad_output * dalpha_dc
        dl_db = grad_output * dalpha_db
        dl_da = grad_output * dalpha_da

        return dl_dc, dl_da, dl_db, None, None



