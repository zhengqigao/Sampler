import sys
sys.path.append("../")

import torch
from sampler.base import score_estimator, mh_sampling
from sampler._common import Distribution, Condistribution
import torch.nn as nn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np


class MultiGauss(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.nn.Parameter(mean)
        self.std = torch.nn.Parameter(std)
        self.dim = len(self.mean)
        self.mul_factor = 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        mean_tensor = torch.zeros_like(self.mean.data).repeat(num_samples, 1)
        std_tensor = torch.ones_like(self.std.data).repeat(num_samples, 1)
        return torch.normal(mean_tensor, std_tensor).reshape(num_samples, self.dim) * self.std + self.mean

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


class MultiGauss2(MultiGauss):
    def __init__(self, mean, std):
        super(MultiGauss2, self).__init__(mean, std)

    def sample(self, num_samples: int) -> torch.Tensor:
        mean_tensor = self.mean.data.repeat(num_samples, 1)
        std_tensor = self.std.data.repeat(num_samples, 1)
        return torch.normal(mean_tensor, std_tensor).reshape(num_samples, self.dim)


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
        return (torch.randn((num_samples, y.shape[0], y.shape[1])) * self.std + y).to(y.device)

    def evaluate_density(self, x: torch.Tensor, y: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        # x is of shape (N,d), y is of shape (M,d)
        # return shape (N,M)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        if in_log:
            return -0.5 * (torch.sum(((x - y) / self.std) ** 2, dim=2) + torch.log(
                2 * torch.pi * self.std * self.std).sum()).to(y.device)
        else:
            return torch.exp(-0.5 * torch.sum(((x - y) / self.std) ** 2, dim=2)).to(y.device) / (
                    torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim * torch.prod(self.std)).to(y.device)


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.fc(x)


class CustomizeDistribution(Distribution):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.model = MLP(in_dim)
        self.mul_factor = None

    def sample(self, num_samples: int) -> torch.Tensor:
        tmp, _ = mh_sampling(num_samples, target=self,
                             transit=ConditionalMultiGauss(torch.ones(self.in_dim)), initial=torch.zeros((1, 2)),
                             burn_in=100)
        # Imagine if inside the implementation of mh_sampling, it requires target.sample() method, then we will have an infinite loop
        # self.sample -> mh_sampling -> target.sample -> mh_sampling ....
        # however, this will never happen. Becasue mh_sampling itself is a sampling function, and its goal is how to sample from the target
        # so it won't use target.sample inside mh_sampling.
        return tmp[:, 0, :]

    def evaluate_density(self, x: torch.Tensor, in_log: bool = True) -> torch.Tensor:
        if in_log:
            return self.model(x).squeeze(1)
        else:
            return torch.exp(self.model(x).squeeze(1))


def run_exp(instance):
    print("-" * 70)

    eval_func = lambda x: x ** 2  # essentially evaluate E_p[x^2] = var + mean^2 = std ** 2 + mean ^ 2

    # by using score_estimator, we can always get gradient
    results = score_estimator(10000, instance, eval_func)
    print("forward result:", results)
    loss = results.sum()  # loss = std[0] ** 2 + std[1] ** 2 + .... std[n-1] ** 2 + mean[0] ** 2 + mean[1] ** 2 + .... mean[n-1] ** 2
    loss.backward()  # dloss/dmean_i = 2 * mean_i, dloss/dstd_i = 2 * std_i. If the parameter is not mean_i or std_i, then it will be every complicated (imagine chain rule)
    for param in instance.parameters():
        print("backward gradient:", param.grad)

    instance.zero_grad()

    # a dummy way to get the gradient, this doesn't always work!
    wrk = instance.sample(10000)
    obj = torch.mean(eval_func(wrk), dim=0)
    print("dummy forward:", obj)
    try:
        loss_dummy = obj.sum()
        loss_dummy.backward()
        for param in instance.parameters():
            print("dummy backward:", param.grad)
    except Exception as e:
        print("dummy backward failed:", e)


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


def run_density_matching_example():
    potential_func = PotentialFunc("potential6")

    # show poential_function
    bound = 4
    x = torch.linspace(-bound, bound, 100)
    y = torch.linspace(-bound, bound, 100)
    xx, yy = torch.meshgrid(x, y)
    grid_data = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)

    value = potential_func(grid_data)

    # scatter them to see the potential on a heatmap
    plt.figure()
    plt.scatter(grid_data[:, 0], grid_data[:, 1], c=torch.exp(-value), cmap='viridis')
    plt.title('golden result')
    plt.savefig("./tmp_golden_result.png")

    module = CustomizeDistribution(2)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.001)
    max_iter, num_sample = 100, 50000
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = score_estimator(num_sample, module, lambda x: (potential_func(x) + module(x, in_log=True)).mean())
        loss.backward()
        optimizer.step()
        print(f"iter {i}, loss: {loss.item()}")

    # show the final result
    plt.figure()
    wrk = module.sample(10000)
    plt.scatter(wrk[:, 0], wrk[:, 1], c=module(wrk).detach().cpu().numpy().reshape(-1), cmap='viridis')
    plt.title("final result learnt by the model using score_estimator")
    plt.savefig("./tmp_final_result.png")

    plt.show()


if __name__ == '__main__':
    mean, std = torch.Tensor([0.0, 1.0]), torch.Tensor([1.0, 2.0])

    # this is the most specical case, because we write it as N(0,1) * std + mean, so we have done reparametrization trick.
    # Thus, the dummy way can have gradients.
    instance = MultiGauss(mean, std)
    run_exp(instance)

    # this is another way to write how to sample from N(mu, std^2). In this way, the mean and std are inside the random generation,
    # thus, the dummy way cannot have gradients.
    instance2 = MultiGauss2(mean, std)
    run_exp(instance2)

    # this is what we are supposed to use the score_esimator in a general nn.Module.
    instance3 = CustomizeDistribution(2)
    run_exp(instance3)

    run_density_matching_example()
