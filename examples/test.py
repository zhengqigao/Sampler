import numpy as np
import torch
from sampler.classical_sampler import importance_sampling, Distribution
from typing import Optional


test_mean = [-1,1, 0.5]

# class multigauss(Distribution):
#     def __init__(self, mean, std):
#         self.dim = len(mean)
#         self.mean = mean
#         self.std = std
#     def sample(self, num_samples: int) -> np.ndarray:
#         return np.random.normal(loc=self.mean, scale=self.std, size=(num_samples, self.dim))
#
#     def __call__(self, x: np.ndarray, in_log: Optional[bool] = True) -> np.ndarray:
#         if in_log:
#             return -0.5 * (np.sum(((x - self.mean) / self.std) ** 2, axis=1) + np.log(2 * np.pi) * self.dim)
#         else:
#             return np.exp(-0.5 * (np.sum(((x - self.mean) / self.std) ** 2, axis=1)) / (np.sqrt(2 * np.pi) ** self.dim * np.prod(self.std)))
#
#
#
# target = multigauss( mean=test_mean, std=[1, 1, 1])
# proposal = multigauss(mean=[0, 0, 0], std=[1, 1, 1])
#
# results = importance_sampling(10000, target, proposal, lambda x: x)
#
# print("Test by numpy:", results)


class MultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(mean)
        self.const = 1.0
    def sample(self, num_samples: int) -> torch.Tensor:
        # PyTorch's distributions.Normal can also be used, but this is straightforward and similar to your numpy approach
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        if in_log:
            return -0.5 * (torch.sum(((x - self.mean) / self.std) ** 2, dim=1) + torch.log(torch.tensor(2 * torch.pi)) * self.dim)
        else:
            # Here, we manually compute the density without the logarithm
            return torch.exp(-0.5 * torch.sum(((x - self.mean) / self.std) ** 2, dim=1)) / (torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim * torch.prod(self.std))



target = MultiGauss(mean=test_mean, std=[1, 1, 1])
proposal = MultiGauss(mean=[0, 0, 0], std=[1, 1, 1])
results = importance_sampling(10000, target, proposal, lambda x: x)
print("Test by pytorch:", results)