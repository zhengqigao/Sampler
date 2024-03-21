import torch
from sampler.base import score_estimator
from sampler._common import Distribution
import torch.nn as nn


# class ExampleModel(torch.nn.Module):
#     def __init__(self, mean, std):
#         super(ExampleModel, self).__init__()
#         self.mean = nn.Parameter(torch.tensor(mean))
#         self.std = nn.Parameter(torch.tensor(std))
#
#     def sample(self, num_samples: int) -> torch.Tensor:
#         return torch.randn((num_samples, 1)) * self.std + self.mean
#
#     def evaluate_density(self, x, in_log):
#         if in_log == False:
#             return torch.exp(-0.5 * ((x - self.mean) / self.std) ** 2) / (
#                         torch.sqrt(torch.tensor(2 * torch.pi)) * self.std)
#         else:
#             return -0.5 * ((x - self.mean) / self.std) ** 2 - torch.log(
#                 torch.sqrt(torch.tensor(2 * torch.pi)) * self.std)
#
#     def forward(self, x, in_log):
#         return self.evaluate_density(x, in_log)


class MultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.nn.Parameter(mean)
        self.std = torch.nn.Parameter(std)
        self.dim = len(self.mean)
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        if in_log:
            return -0.5 * (
                torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
                + torch.log(2 * torch.pi * self.std * self.std).sum()
            )
        else:
            return torch.exp(
                -0.5 * torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
            ) / (
                torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim
                * torch.prod(self.std)
            )

mean, std = torch.Tensor([0.0, 1.0]), torch.Tensor([1.0, 2.0])
instance = MultiGauss(mean, std)
eval_func = lambda x: x ** 2 # essentially evaluate E_p[x^2] = var + mean^2 = std ** 2 + mean ^ 2
results = score_estimator(10000, instance, eval_func)
print("forward result:", results)
print("before backward:", instance.mean.grad, instance.std.grad)

loss = results.sum() # loss = std[0] ** 2 + std[1] ** 2 + .... std[n-1] ** 2 + mean[0] ** 2 + mean[1] ** 2 + .... mean[n-1] ** 2
loss.backward() # dloss/dmean_i = 2 * mean_i, dloss/dstd_i = 2 * std_i
print("backward result", instance.mean.grad, instance.std.grad)
