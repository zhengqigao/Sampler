from typing import Optional, Union, Tuple, Any
import torch
import torch.nn as nn
from .._common import BiProbTrans, Distribution, Func
import warnings
import torch.nn.init as init
import math

class RadialFlow(BiProbTrans):
    r"""
    The Radial flow model described in ..[Rezende2015]. The invertibility conditions described in their Appendix are implemented.

    """

    def __init__(self, dim: int,
                 num_trans: int,
                 p_base: Optional[Distribution] = None):

        super().__init__(p_base=p_base)
        self.dim = dim
        self.num_trans = num_trans

        self.z0 = nn.Parameter(torch.empty(num_trans, dim))
        self.alpha = nn.Parameter(torch.empty(num_trans))
        self.beta = nn.Parameter(torch.empty(num_trans))

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.z0, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.z0)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.alpha, -bound, bound)
        init.uniform_(self.beta, -bound, bound)


    def reparametrize_beta(self) -> torch.Tensor:
        beta = -self.alpha + nn.functional.softplus(self.beta)
        return beta

    def forward(self, x: torch.Tensor, log_prob: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[
        torch.Tensor, torch.Tensor]:

        beta = self.reparametrize_beta()
        for i in range(self.num_trans):
            z0, alpha, beta_i = self.z0[i], self.alpha[i], beta[i]
            diff = x - z0
            r = torch.norm(diff, dim=1)

            h, dh = 1 / (alpha + r), -1 / (alpha + r) ** 2

            x = x + beta_i * h.unsqueeze(-1) * diff
            log_prob = log_prob - (self.dim - 1) * torch.log(1 + beta_i * h) - torch.log(1 + beta_i * h + beta_i * dh * r)

        return x, log_prob

    def backward(self, z: torch.Tensor, log_prob: Optional[Union[float, torch.Tensor]] = 0.0) -> Tuple[
        torch.Tensor, torch.Tensor]:
        beta = self.reparametrize_beta()
        for i in range(self.num_trans-1, -1,-1):
            z0, alpha, beta_i = self.z0[i], self.alpha[i], beta[i]
            diff = z - z0
            diff_abs = torch.norm(diff, dim=1)

            a, b, c = 1, alpha + beta_i, - diff_abs
            r = 1.0 / (2 * a) * (-b + torch.sqrt(b ** 2 - 4 * a * c))

            z_hat = diff / (r.unsqueeze(-1) * (1 + beta_i / (alpha + r.unsqueeze(-1))))

            z = z0 + r.unsqueeze(-1) * z_hat + beta_i * r.unsqueeze(-1) * z_hat / (alpha + r.unsqueeze(-1))

            h, dh = 1 / (alpha + r), -1 / (alpha + r) ** 2
            log_prob = log_prob + (self.dim - 1) * torch.log(1 + beta_i * h) + torch.log(1 + beta_i * h + beta_i * dh * r)

        return z, log_prob