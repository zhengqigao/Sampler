from sampler.base import *
from sampler._common import Distribution
from test_common_helper import PotentialFunc

test_mean = [-1, 1, 0.5]


class MultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)
        self.std = std if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)
        self.dim = len(mean)
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (torch.sum(((x - self.mean) / self.std) ** 2, dim=1) + torch.log(
            2 * torch.pi * self.std * self.std).sum())


target = MultiGauss(mean=test_mean, std=[1, 1, 1])
target.mul_factor = None

try:
    proposal = MultiGauss(mean=[0, 0, 0], std=[1, 1, 1])
    results, info = rejection_sampling(10000, target, proposal, k=1000)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
# Not always successful.
# Because both target and proposal are Gaussian,
# thus proposal std should be greater than target std,
# otherwise k*proposal could never cover target.

# 3 specially chosen proposals with larger std:
try:
    proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
    results, info = rejection_sampling(10000, target, proposal, k=40)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
try:
    proposal = MultiGauss(mean=[0, 0, 0], std=[2, 2, 2])
    results, info = rejection_sampling(10000, target, proposal, k=155)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
try:
    proposal = MultiGauss(mean=[0, 0, 0], std=[3, 3, 3])
    results, info = rejection_sampling(10000, target, proposal, k=900)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    
print("=====================================")

# exceptional cases for k
# non-positive k rejected
try:
    proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
    results, info = rejection_sampling(10000, target, proposal, k=0)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
try:
    proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
    results, info = rejection_sampling(10000, target, proposal, k=-100)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    
# float k accepted
try:
    proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
    results, info = rejection_sampling(10000, target, proposal, k=40.5)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)

# # k=inf seems dead loop
# try:
#     proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
#     results, info = rejection_sampling(10000, target, proposal, k=torch.inf)
#     print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
# except Exception as e:
#     print(e)

# k=nan accepted (why?)
# Mean: tensor([nan, nan, nan])   Rejection rate: 0.0   Size: torch.Size([0, 3])
try:
    proposal = MultiGauss(mean=[0, 0, 0], std=[1.2, 1.2, 1.2])
    results, info = rejection_sampling(10000, target, proposal, k=torch.nan)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)

print("=====================================")

# Custom-defined target (2D only)
class CustomTarget1(Distribution):
    def __init__(self):
        super().__init__()
    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        return torch.exp(-PotentialFunc("potential1")(x))

target1 = CustomTarget1()
target1.mul_factor = None
proposal1 = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
try:
    results, info = rejection_sampling(10000, target1, proposal1, k=200)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)

# Another custom-defined target (2D only)
class CustomTarget2(Distribution):
    def __init__(self):
        super().__init__()
    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        return torch.exp(-PotentialFunc("potential6")(x))

target2 = CustomTarget2()
target2.mul_factor = None
proposal2 = MultiGauss(mean=[0, 0], std=[1.5, 1.5])
try:
    results, info = rejection_sampling(10000, target2, proposal2, k=200)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)

# Another custom-defined target (2D only)
class CustomTarget3(Distribution):
    def __init__(self):
        super().__init__()
    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        return torch.exp(-PotentialFunc("potential7")(x))

target3 = CustomTarget3()
target3.mul_factor = None
proposal3 = MultiGauss(mean=[0, 0], std=[2, 2])
try:
    results, info = rejection_sampling(10000, target3, proposal3, k=200)
    print(f"Mean: {torch.mean(results, dim=0)}\tRejection rate: {info['rejection_rate']}\tSize: {results.shape}")
except Exception as e:
    print(e)
    