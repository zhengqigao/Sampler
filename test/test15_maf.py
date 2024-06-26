import sys
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../"))
import numpy as np
from sampler.model import MAF, MAFLayer, MADE
from test_common_helper import Feedforward, MultiGauss, PotentialFunc
from sampler._common import Distribution
from sampler.base import importance_sampling
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn import datasets
from sampler.functional import KLDenLoss, KLGenLoss, ScoreDenLoss


def test_maf():
    dim = 4
    num_trans = 10
    mg = MultiGauss(mean=[0] * dim, std=[1] * dim)
    flowtransform = MAF(dim=dim, num_trans=num_trans,
                        made=nn.ModuleList(MADE(hidden_dims=[dim, 20, 2 * dim]) for _ in range(num_trans)),
                        p_base=mg)
    print(flowtransform.p_base)
    x = torch.rand(100, dim)
    x.requires_grad = True
    x_, log_prob = flowtransform.backward(*flowtransform.forward(x, 0))
    print(f"diff = {torch.max(torch.abs(x - x_)):.3e}, log_prob = {log_prob.abs().max():.3e}")
    print(x[0,])


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_trans = 30
    dim = 2
    mg = MultiGauss(mean=[0] * dim, std=[1] * dim)
    module = MAF(dim=dim, num_trans=num_trans,
                 made=nn.ModuleList(MADE(hidden_dims=[dim, 20, 2 * dim]) for _ in range(num_trans)),
                 p_base=mg).to(device)

    optimizer = torch.optim.Adam(module.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    max_iter = 10
    loss_list = []
    batch_size = 1000
    criterion1 = KLDenLoss(log_p=lambda x: -potential_func(x))

    for i in range(max_iter):
        loss = criterion1(module, batch_size)

        loss_list.append(loss.item())
        if torch.isnan(loss).any() or i == max_iter - 1:
            plt.figure()
            plt.plot(loss_list)
            plt.show()
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"iter {i}, loss: {loss.item()}")

    plt.figure()
    samples, log_prob = module.sample(50000)
    samples = samples.detach().cpu().numpy()
    log_prob = log_prob.detach().cpu().numpy()
    plt.scatter(samples[:, 0], samples[:, 1], c=np.exp(log_prob).reshape(-1),
                cmap='viridis')
    plt.colorbar()
    plt.title('learnt module samples')
    plt.xlim(-bound, bound)
    plt.ylim(-bound, bound)
    # plt.figure()
    # plt.scatter(grid_data[:, 0], grid_data[:, 1], c=torch.exp(module.log_prob(grid_data)[1]).detach().numpy(),
    #             cmap='viridis')
    # plt.colorbar()
    # plt.title('learned module distribution')
    plt.show()


def run_generation_example(plot_or_save='plot', device='cpu'):
    device = torch.device(device)

    num_trans = 30
    dim = 2
    mg = MultiGauss(mean=[0] * dim, std=[1] * dim)
    module = MAF(dim=dim, num_trans=num_trans,
                 made=nn.ModuleList(MADE(hidden_dims=[dim, 20, 2 * dim]) for _ in range(num_trans)),
                 p_base=mg).to(device)

    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    num_steps = 10
    criterion = KLGenLoss()
    loss_list = []
    for i in range(num_steps):
        z, _ = datasets.make_moons(n_samples=1000, noise=0.1)
        z = torch.Tensor(z).to(device)
        loss = criterion(module, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"iter {i}, loss: {loss.item()}")
        scheduler.step()
        loss_list.append(loss.item())
    # show the generated samples
    samples, log_prob = module.sample(10000)
    samples = samples.cpu().detach().numpy()
    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.title("generated samples")
    plt.xlim(-1.5, 2.5)
    plt.ylim(-1.5, 1.5)
    if plot_or_save == 'save':
        plt.savefig('tmp_generated.png')

    plt.figure()
    x, _ = datasets.make_moons(n_samples=1000, noise=0.1)
    plt.scatter(x[:, 0], x[:, 1])
    plt.title("real samples")
    plt.xlim(-1.5, 2.5)
    plt.ylim(-1.5, 1.5)
    plt.figure()
    plt.plot(loss_list)
    plt.title('loss list')

    if plot_or_save == 'plot':
        plt.show()


# test_maf()
# run_density_matching_example()
run_generation_example('plot','cpu')
