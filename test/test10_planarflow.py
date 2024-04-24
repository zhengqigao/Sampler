import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
import sys
sys.path.append('../')
from sampler._common import BiProbTrans, Distribution
from sampler.model.planarflow import PlanarFlow
from test_common_helper import Feedforward, MultiGauss, PotentialFunc
import matplotlib.pyplot as plt
from sampler.functional import KLDenLoss

def test_planar_flow():
    dim = 4
    mg = MultiGauss(mean=[0] * dim, std=[1] * dim)
    flowtransform = PlanarFlow(dim=dim, num_trans=10, p_base= mg)
    print(flowtransform.p_base)
    x = torch.rand(10, dim)
    x.requires_grad = True
    x_, log_prob = flowtransform.backward(*flowtransform.forward(x, 0))
    print(f"diff = {torch.max(torch.abs(x - x_)):.3e}, log_prob = {log_prob.abs().max():.3e}")

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

    num_trans = 12
    dim = 2
    mg = MultiGauss(mean=[0] * dim, std=[1] * dim)
    module = PlanarFlow(dim=dim, num_trans=num_trans, p_base= mg)

    optimizer = torch.optim.Adam(module.parameters(), lr=0.0001)
    max_iter = 500
    loss_list = []
    batch_size = 1000
    criterion1 = KLDenLoss(log_p = lambda x: -potential_func(x))

    for i in range(max_iter):
        loss = criterion1(module, batch_size)

        loss_list.append(loss.item())
        if torch.isnan(loss).any() or i == max_iter - 1:
            plt.figure()
            plt.plot(loss_list)
            plt.show()
            break
        optimizer.zero_grad()
        loss.backward() #
        optimizer.step()
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
    plt.figure()
    plt.scatter(grid_data[:, 0], grid_data[:, 1], c=torch.exp(module.log_prob(grid_data)[1]).detach().numpy(),
                cmap='viridis')
    plt.colorbar()
    plt.title('learned module distribution')
    plt.show()


test_planar_flow()
run_density_matching_example()