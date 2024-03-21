import torch
from sampler.base import score_estimator
from sampler._common import Distribution
import torch.nn as nn


class ExampleModel(torch.nn.Module):
    def __init__(self, mean, std):
        super(ExampleModel, self).__init__()
        self.mean = nn.Parameter(torch.tensor(mean))
        self.std = nn.Parameter(torch.tensor(std))

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, 1)) * self.std + self.mean

    def evaluate_density(self, x, in_log):
        if in_log == False:
            return torch.exp(-0.5 * ((x - self.mean) / self.std) ** 2) / (
                        torch.sqrt(torch.tensor(2 * torch.pi)) * self.std)
        else:
            return -0.5 * ((x - self.mean) / self.std) ** 2 - torch.log(
                torch.sqrt(torch.tensor(2 * torch.pi)) * self.std)

    def forward(self, x, in_log):
        return self.evaluate_density(x, in_log)


mean, std = 0.0, 2.0
instance = ExampleModel(mean, std)
eval_func = lambda x: (x - 0.0) ** 2
results = score_estimator(10000, instance, eval_func)
print("forward result:", results)
results.backward()
print("backward result", instance.mean.grad, instance.std.grad)
