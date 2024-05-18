import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import sys
import os
import math

sys.path.append(os.path.abspath("../"))
from sampler._common import Distribution, Condistribution


class Feedforward(nn.Module):
    def __init__(self, hidden_dims, activation='leakyrelu'):
        super(Feedforward, self).__init__()

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)])

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Invalid activation function: {activation}")

    def forward(self, x):

        for i in range(len(self.hidden_layers) - 1):
            x = self.activation(self.hidden_layers[i](x))
        x = self.hidden_layers[-1](x)
        return x


# these are copied from a paper:  https://arxiv.org/pdf/1505.05770.pdf
# For precise description, say we denote: distribution_value = exp(density) = exp(-potential) In the
# evalulate_density function defined in our Distribution class, we need to provide `distribution_value` when
# in_log=False, and `density` when in_log=True. However, here the PotentialFunc actually returns the `potential` not the `density`.

class PotentialFunc(object):
    def __init__(self, name: str):
        self.potential = getattr(self, name)

    def __call__(self, z: torch.Tensor, cal_type=1) -> torch.Tensor:
        if cal_type == 1:
            if z.shape[1] != 2:
                raise ValueError(f"Input shape {z.shape} is not supported")
            else:
                return self.potential(z)
        else:
            raise NotImplementedError(f"Cal type {cal_type} is not implemented")

    def potential1(self, z: torch.Tensor) -> torch.Tensor:
        z1, z2 = z[:, 0], z[:, 1]
        t1 = 0.5 * ((torch.norm(z, dim=1) - 2) / 0.4) ** 2
        wrk1 = torch.exp(-0.5 * ((z1 - 2) / 0.6) ** 2)
        wrk2 = torch.exp(-0.5 * ((z1 + 2) / 0.6) ** 2)
        t2 = torch.log(wrk1 + wrk2)
        return t1 - t2

    def potential2(self, z: torch.Tensor) -> torch.Tensor:
        z1, z2 = z[:, 0], z[:, 1]
        w1 = torch.sin(2 * np.pi * z1 / 4)
        return 0.5 * ((z2 - w1) / 0.4) ** 2

    def potential3(self, z: torch.Tensor) -> torch.Tensor:
        z1, z2 = z[:, 0], z[:, 1]
        w1 = torch.sin(2 * np.pi * z1 / 4)
        w2 = 3 * torch.exp(-0.5 * ((z1 - 1) / 0.6) ** 2)

        wrk1 = torch.exp(-0.5 * ((z2 - w1) / 0.35) ** 2)
        wrk2 = torch.exp(-0.5 * ((z2 - w1 + w2) / 0.35) ** 2)
        return -torch.log(wrk1 + wrk2)

    def potential4(self, z: torch.Tensor) -> torch.Tensor:
        z1, z2 = z[:, 0], z[:, 1]
        w1 = torch.sin(2 * np.pi * z1 / 4)
        w3 = 3 * torch.nn.functional.sigmoid((z1 - 1) / 0.3)

        wrk1 = torch.exp(-0.5 * ((z2 - w1) / 0.4) ** 2)
        wrk2 = torch.exp(-0.5 * ((z2 - w1 + w3) / 0.35) ** 2)
        return -torch.log(wrk1 + wrk2)

    def potential5(self, z: torch.Tensor) -> torch.Tensor:
        p_z0 = torch.distributions.MultivariateNormal(
            loc=torch.zeros(2, ).to(z.device),
            covariance_matrix=torch.diag(torch.ones(2, ).to(z.device))
        )
        return -p_z0.log_prob(z)

    def potential6(self, z: torch.Tensor) -> torch.Tensor:
        # Customize your mixture of Gaussians logic here
        gaussian1 = torch.distributions.MultivariateNormal(
            loc=torch.tensor([-1.5, 0.0]).to(z.device),
            covariance_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to(z.device)
        )

        gaussian2 = torch.distributions.MultivariateNormal(
            loc=torch.tensor([1.5, 0.0]).to(z.device),
            covariance_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to(z.device)
        )

        weight1 = 0.5  # Adjust the weights as needed
        weight2 = 0.5

        return -torch.log(weight1 * torch.exp(gaussian1.log_prob(z)) + weight2 * torch.exp(gaussian2.log_prob(z)))

    def potential7(self, z: torch.Tensor) -> torch.Tensor:
        # Customize your mixture of Gaussians logic here
        gaussian1 = torch.distributions.MultivariateNormal(
            loc=torch.tensor([-1.8, -1]).to(z.device),
            covariance_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to(z.device)
        )

        gaussian2 = torch.distributions.MultivariateNormal(
            loc=torch.tensor([1.8, -1]).to(z.device),
            covariance_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to(z.device)
        )

        gaussian3 = torch.distributions.MultivariateNormal(
            loc=torch.tensor([0.0, 1.2]).to(z.device),
            covariance_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to(z.device)
        )

        weight1 = 1 / 3  # Adjust the weights as needed
        weight2 = 1 / 3
        weight3 = 1 / 3

        return -torch.log(weight1 * torch.exp(gaussian1.log_prob(z)) +
                          weight2 * torch.exp(gaussian2.log_prob(z)) +
                          weight3 * torch.exp(gaussian3.log_prob(z)))


class DensityFunc(object):
    def __init__(self, name: str):
        self.name = name

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return -PotentialFunc(self.name)(z)


def PlotSamples(target, results, info):
    x = torch.linspace(-5, 5, 100)
    y = torch.linspace(-5, 5, 100)
    xx, yy = torch.meshgrid(x, y)
    grid_data = torch.cat((yy.reshape(-1, 1), xx.reshape(-1, 1)), dim=1)
    value = target(grid_data)
    z = torch.exp(value).reshape(100,-1)
    plt.figure()
    plt.title(f"Mean: {torch.mean(results, dim=0)}\nRejection rate: {info['rejection_rate']}")
    plt.pcolormesh(x, y, z[:-1, :-1], cmap='summer')
    plt.colorbar()
    plt.scatter(results[:, 0], results[:, 1], marker='.', color='red', alpha=0.1)
    plt.show()


class TensorizedMultiGauss(Distribution):
    def __init__(self, mean, std, device=torch.device("cpu")):
        super().__init__()
        self.mean = mean.to(device) if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32).to(
            device)
        self.std = std.to(device) if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32).to(
            device)
        self.dim = self.mean.shape
        self.device = device
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return (torch.randn((num_samples, *self.dim)).to(self.device) * self.std + self.mean)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
                torch.sum(((x - self.mean) / self.std) ** 2, dim=tuple(range(1, len(self.dim) + 1)))
                + torch.log(2 * torch.pi * self.std * self.std).sum()
        )


class TensorizedConditionalMultiGauss(Condistribution):
    def __init__(self, std, device=torch.device("cpu")):
        super().__init__()
        # self.mean = mean.to(device) if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32).to(device)
        self.std = std.to(device) if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32).to(
            device)
        # self.std = torch.tensor(std, dtype=torch.float32)
        self.dim = len(std)
        self.mul_factor = 1.0
        self.device = device

    def sample(self, num_samples: int, y) -> torch.Tensor:
        # y has shape (m, d)
        # return shape (num_samples, m, d) with y as the mean
        assert len(y.shape) == 2 and y.shape[1] == self.dim
        self.y = y.to(self.device) if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32).to(
            self.device)
        return (torch.randn((num_samples, y.shape[0], y.shape[1])).to(self.device)) * self.std + y

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x is of shape (N,d), y is of shape (M,d)
        # return shape (N,M)
        x = x.unsqueeze(1).to(self.device)
        y = y.unsqueeze(0).to(self.device)
        return -0.5 * (
                torch.sum(((x - y) / self.std) ** 2, dim=2) + torch.log(2 * torch.pi * self.std * self.std).sum())


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

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x is of shape (N,d), y is of shape (M,d)
        # return shape (N,M)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return -0.5 * (
                torch.sum(((x - y) / self.std) ** 2, dim=2) + torch.log(2 * torch.pi * self.std * self.std).sum())


''' 
2-dimension Examples of Gibbs Sampling from [Peng2020gibbs] 
'''
class CondGaussGamma(Condistribution):
    def __init__(self, data, w):
        # data has shpae (n, 1)
        # w is a scalar
        super().__init__()
        self.num_data = data.shape[0]
        self.sum_data = torch.sum(data)
        self.w = 1/w**2
        self.dim = 1
        self.mul_factor = 1.0

    def sample(self, num_samples: int, y) -> torch.Tensor:
        # y has shape (m, 1)
        # return shape (num_samples, m, 1)
        # print("y: {}".format(y))
        mean = y / (y * self.num_data + self.w) * self.sum_data
        # print("mean: {}".format(mean))
        std = 1/math.sqrt(y * self.num_data + self.w)
        # print("std: {}".format(std))
        assert len(y.shape) == 2 and y.shape[1] == self.dim
        return torch.randn((num_samples, y.shape[0], y.shape[1])) * std + mean

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x is of shape (N,1), y is of shape (M,1)
        # return shape (N,M)
        mean = y / (y * self.num_data + self.w) * self.sum_data
        std = 1/math.sqrt(y * self.num_data + self.w)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return -0.5 * (
                torch.sum(((x - y) / std) ** 2, dim=1) + torch.log(2 * torch.pi * std * std).sum())


class CondGammaGauss(Condistribution):
    def __init__(self, data, alpha, beta):
        # data has shpae (n, 1)
        # w is a scalar
        super().__init__()
        self.num_data = data.shape[0]
        self.sum_data = torch.sum(data)
        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.dim = 1
        self.mul_factor = 1.0

    def sample(self, num_samples: int, y) -> torch.Tensor:
        # y has shape (m, 1)
        # return shape (num_samples, m, 1)
        t = self.alpha + self.num_data / 2
        theta = self.beta + torch.sum((self.data - y) ** 2) / 2
        #print("theta: {}".format(theta))
        assert len(y.shape) == 2 and y.shape[1] == self.dim
        gamma = torch.rand((int(t), num_samples, y.shape[0], y.shape[1]))
        while torch.isinf(torch.log(gamma).sum(0)).any():
            gamma = torch.rand((int(t), num_samples, y.shape[0], y.shape[1]))
        #print("gamma: {}".format(gamma))
        sample_value = - torch.log(gamma).sum(0) / theta
        #print(gamma.shape)
        # print("sample_value: {}".format(sample_value))
        return sample_value

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x is of shape (N,1), y is of shape (M,1)
        # return shape (N,M)
        t = self.alpha + self.num_data / 2
        theta = self.beta + torch.sum((self.data - y) ** 2) / 2
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return -x * theta + (t - 1) * torch.log(x)


class CorMultiGauss2D(Distribution):
    def __init__(self, mean, std, rho):
        super().__init__()
        # 2-dimensional
        self.mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)
        self.rho = rho
        self.dim = len(self.mean)
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        Z = torch.randn((num_samples, self.dim))
        Z[:][0] = self.std[:][0] * Z[:][0]
        Z[:][1] = self.std[:][1] * (Z[:][0]*self.rho + Z[:][1]*math.sqrt(1-self.rho**2))
        return  Z + self.mean

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
                torch.sum(((x - self.mean) / self.std) ** 2, dim=1)
                + torch.log(2 * torch.pi * self.std * self.std).sum()
        )


class CondGaussGauss1D(Condistribution):
    def __init__(self, mean, std, mean_cond, std_cond, rho):
        super().__init__()
        self.mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)
        self.std = std if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)
        self.mean_cond = mean_cond if isinstance(mean_cond, torch.Tensor) else torch.tensor(mean_cond, dtype=torch.float32)
        self.std_cond = std_cond if isinstance(std_cond, torch.Tensor) else torch.tensor(std_cond, dtype=torch.float32)
        self.rho = rho
        self.dim = len(std)
        self.mul_factor = 1.0

    def sample(self, num_samples: int, y) -> torch.Tensor:
        # y has shape (m, d)
        # return shape (num_samples, m, d) with y as the mean
        assert len(y.shape) == 2 and y.shape[1] == self.dim
        new_mean = self.mean + self.rho * self.std * (y - self.mean_cond) / self.std_cond
        new_std = self.std * math.sqrt(1-self.rho**2)
        return torch.randn((num_samples, y.shape[0], y.shape[1])) * new_std + new_mean

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x is of shape (N,d), y is of shape (M,d)
        # return shape (N,M)
        new_mean = self.mean + self.rho * self.std * (self.y - self.mean_cond) / self.std_cond
        new_std = self.std * math.sqrt(1-self.rho**2)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return -0.5 * (
                torch.sum(((x - new_mean) / new_std) ** 2, dim=1) + torch.log(2 * torch.pi * new_std * new_std).sum())


class BlockCondGaussGauss(Condistribution):
    def __init__(self, mean_a, mean_b, sigma_aa, sigma_bb, sigma_ab, sigma_ba):
        super().__init__()
        self.mean_a = mean_a if isinstance(mean_a, torch.Tensor) else torch.tensor(mean_a, dtype=torch.float32)
        self.mean_b = mean_b if isinstance(mean_b, torch.Tensor) else torch.tensor(mean_b, dtype=torch.float32)
        self.sigma_aa = sigma_aa if isinstance(sigma_aa, torch.Tensor) else torch.tensor(sigma_aa, dtype=torch.float32)
        self.sigma_bb = sigma_bb if isinstance(sigma_bb, torch.Tensor) else torch.tensor(sigma_bb, dtype=torch.float32)
        self.sigma_ab = sigma_ab if isinstance(sigma_ab, torch.Tensor) else torch.tensor(sigma_ab, dtype=torch.float32)
        self.sigma_ba = sigma_ba if isinstance(sigma_ba, torch.Tensor) else torch.tensor(sigma_ba, dtype=torch.float32)
        self.dim_a = len(mean_a)
        self.dim_b = len(mean_b)
        self.mul_factor = 1.0

    def sample(self, num_samples: int, y) -> torch.Tensor:
        # y has shape (m, d)
        # return shape (num_samples, m, D-d)
        #print(f"y: {y}\ny.shape: {y.shape}\ndim_b: {self.dim_b}\nmean_b: {self.mean_b}\nlen(mean_b): {len(self.mean_b)}")
        assert len(y.shape) == 2 and y.shape[1] == self.dim_b
        #print(f"mean_bias: {torch.mm(torch.mm(self.sigma_ab, torch.pinverse(self.sigma_bb)), torch.t(y - self.mean_b))}")
        new_mean = self.mean_a + torch.t(torch.mm(torch.mm(self.sigma_ab, torch.pinverse(self.sigma_bb)), torch.t(y - self.mean_b)))
        new_std = self.sigma_aa - torch.mm(torch.mm(self.sigma_ab, torch.pinverse(self.sigma_bb)), self.sigma_ba)
        #print(f"new_mean: {new_mean}\nnew_std: {new_std}\ntorch.randn((num_samples, y.shape[0], self.dim_a)): {torch.randn((num_samples, y.shape[0], self.dim_a))}")
        samples = torch.matmul(torch.randn((num_samples, y.shape[0], self.dim_a)), new_std) + new_mean
        #print(f"y.shape[0]: {y.shape[0]}")
        #print(f"samples: {samples}\nsamples.shape: {samples.shape}")
        return samples

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x is of shape (N,d), y is of shape (M,d)
        # return shape (N,M)
        new_mean = self.mean_a + self.sigma_ab * torch.pinverse(self.sigma_bb) * (y - self.mean_b)
        new_std = self.sigma_aa - self.sigma_ab * torch.pinverse(self.sigma_bb) * self.sigma_ba
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return -0.5 * (
                torch.sum(((x - new_mean) / new_std) ** 2, dim=1) + torch.log(2 * torch.pi * new_std * new_std).sum())


class UnconditionalMultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)
        self.std = std if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)
        self.dim = len(self.mean)
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.dim)) * self.std + self.mean

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
                torch.sum(((x - self.mean.to(x.device)) / self.std.to(x.device)) ** 2, dim=1)
                + torch.log(2 * torch.pi * self.std * self.std).sum()
        )


MultiGauss = UnconditionalMultiGauss

if __name__ == '__main__':
    func_list = ['potential1',
                 'potential2',
                 'potential3',
                 'potential4',
                 'potential5',
                 'potential6',
                 'potential7', ]

    for i in range(len(func_list)):
        key = func_list[i]
        potential_func = PotentialFunc(key)

        bound = 4
        # generate grid data in [-3,3] * [-3,3]
        x = torch.linspace(-bound, bound, 100)
        y = torch.linspace(-bound, bound, 100)
        xx, yy = torch.meshgrid(x, y)
        grid_data = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)

        value = potential_func(grid_data)

        # scatter them to see the potential on a heatmap
        plt.figure()
        plt.title(key)
        plt.scatter(grid_data[:, 0], grid_data[:, 1], c=torch.exp(-value), cmap='viridis')
        plt.colorbar()
        plt.show()
