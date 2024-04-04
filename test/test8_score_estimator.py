import sys
sys.path.append("../")

import torch
from sampler.base import score_estimator, mh_sampling
from sampler._common import Distribution, Condistribution
import torch.nn as nn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
from test_common_helper import PotentialFunc

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
                    * torch.prod(self.std * self.std)
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
                    torch.sqrt(torch.tensor(2 * torch.pi)) ** self.dim * torch.prod(self.std * self.std)).to(y.device)


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
        tmp, info = mh_sampling(num_samples, target=self,
                             transit=ConditionalMultiGauss(torch.ones(self.in_dim)), initial=torch.zeros((1, 2)),
                             burn_in=1000)
        print(f"acceptance rate: {info['acceptance_rate']}")
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
    plt.savefig("./test/tmp_golden_result.png")

    module = CustomizeDistribution(2)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.001)
    max_iter, num_sample = 20, 10000
    for i in range(max_iter):
        if i % 2 == 0:
            # show the final result
            # plt.figure()
            # wrk = module.sample(50000)
            # plt.scatter(wrk[:, 0], wrk[:, 1], c=module(wrk, in_log = False).detach().cpu().numpy().reshape(-1), cmap='viridis')
            # plt.colorbar()
            # plt.title(f" i={i} Samples of model using MH")
            # plt.figure()
            # plt.scatter(grid_data[:, 0], grid_data[:, 1], c=module(grid_data, in_log=False).detach().cpu().numpy(), cmap='viridis')
            # plt.title(f" i={i} Potential of model")
            # plt.colorbar()
            # plt.savefig("./test/tmp_final_result.png")

            plt.figure()
            wrk = module.sample(50000)
            wrk2 = wrk.detach().cpu().numpy()
            hist, xedges, yedges = np.histogram2d(wrk2[:, 0], wrk2[:, 1], bins=50)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            # Plot 2D histogram
            plt.imshow(hist.T, extent=extent, origin='lower', cmap='viridis')
            plt.colorbar(label='Frequency')
            plt.title("2D Histogram of Samples")
            plt.xlabel("X axis")
            plt.ylabel("Y axis")

            plt.figure()
            scatter = plt.scatter(grid_data[:, 0], grid_data[:, 1],
                                     c=module(grid_data, in_log=False).detach().cpu().numpy(), cmap='viridis')
            plt.title("Potential of model")
            plt.xlabel("X axis")
            plt.ylabel("Y axis")

            # Add colorbar to the second subplot
            cbar = plt.colorbar(scatter)
            cbar.set_label('Potential')

            plt.tight_layout()
            plt.show()


        optimizer.zero_grad()
        loss = score_estimator(num_sample, module, lambda x: (potential_func(x) + module(x, in_log=True)).mean())
        loss.backward()
        optimizer.step()
        print(f"iter {i}, loss: {loss.item()}")

def run_mh_example():
    potential_func = PotentialFunc("potential6")
    tmp, _ = mh_sampling(10000, target=lambda x, in_log: -potential_func(x, True), transit=ConditionalMultiGauss(torch.ones(2)), initial=torch.zeros((1, 2)), burn_in=1000)
    plt.figure()
    plt.scatter(tmp[:, 0, 0], tmp[:, 0, 1], s=1)
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

    run_mh_example()
    # run_density_matching_example()

