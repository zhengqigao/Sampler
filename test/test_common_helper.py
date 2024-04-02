import torch
import numpy as np
import matplotlib.pyplot as plt

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
