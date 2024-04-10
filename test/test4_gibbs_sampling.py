import torch
from sampler._common import Condistribution


class ConditionalMultiGauss(Condistribution):
    def __init__(self, std):
        super().__init__()
        self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(std)
        self.mul_factor = 1.0

    def sample(self, num_samples: int, y) -> torch.Tensor:
        # y has shape (m, d)
        # return shape (num_samples, m, d) with y as the mean
        assert len(y.shape) == 2 and y.shape[1] == self.dim
        return torch.randn((num_samples, y.shape[0], y.shape[1])) * self.std + y

    def evaluate_density(self, x: torch.Tensor, y: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        # x is of shape (N,d), y is of shape (M,d)
        # return shape (N,M)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        if in_log:
            return -0.5 * (torch.sum(((x - y) / self.std) ** 2, dim=2) + torch.log(2 * torch.pi * self.std * self.std).sum())
        else:
            return torch.exp(-0.5 * torch.sum(((x - y) / self.std) ** 2, dim=2)) / (
                    torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim * torch.prod(self.std))


## TODO: use an example of joint Gaussian, so that we know all conditional distributions are Gaussian. Test Gibbs sampling on this example.

