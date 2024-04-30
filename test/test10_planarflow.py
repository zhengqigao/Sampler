import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
import sys
sys.path.append('../')
from sampler._common import BiProbTrans, Distribution
from sampler.model.planarflow import PlanarFlow, AlphaSolver
from test_common_helper import Feedforward, MultiGauss, PotentialFunc
import matplotlib.pyplot as plt
from sampler.functional import KLDenLoss, KLGenLoss
import numpy as np
from sklearn import datasets

def test_planar_flow():
    dim = 4
    mg = MultiGauss(mean=[0] * dim, std=[1] * dim)
    flowtransform = PlanarFlow(dim=dim, num_trans=20, p_base= mg)
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

    num_trans = 32
    dim = 2
    mg = MultiGauss(mean=[0] * dim, std=[1] * dim)
    module = PlanarFlow(dim=dim, num_trans=num_trans, p_base= mg).to(device)

    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)

    max_iter = 2000
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
        loss.backward()
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
    # plt.figure()
    # plt.scatter(grid_data[:, 0], grid_data[:, 1], c=torch.exp(module.log_prob(grid_data)[1]).detach().numpy(),
    #             cmap='viridis')
    # plt.colorbar()
    # plt.title('learned module distribution')
    plt.show()


def run_generation_example(plot_or_save = 'plot', device='cpu'):
    device = torch.device(device)
    num_trans = 40
    dim = 2
    mg = MultiGauss(mean=[0] * dim, std=[1] * dim)
    module = PlanarFlow(dim=dim, num_trans=num_trans, p_base= mg, alpha_threshold=1e-10).to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
    num_steps = 1000
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
    if plot_or_save == 'save':
        plt.savefig('tmp_generated.png')

    plt.figure()
    x, _ = datasets.make_moons(n_samples=1000, noise=0.1)
    plt.scatter(x[:, 0], x[:, 1])
    plt.title("real samples")

    plt.figure()
    plt.plot(loss_list)
    plt.title('loss list')

    if plot_or_save == 'plot':
        plt.show()

def test_alpha_solve():
    eps = 1e-4  # Small perturbation

    alpha_iter = 10000
    alpha_threshold = 1e-13

    # Function to compute the forward pass
    def compute_output(c, a, b):
        alpha = AlphaSolver.apply(c, a, b, alpha_iter, alpha_threshold)
        return (alpha ** 2).sum()

    c = torch.rand((100,), requires_grad=True)
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([0.5], requires_grad=True)


    output = compute_output(c, a, b)
    output.backward()
    print("Analytical dl_dc:", c.grad)
    print("Analytical dl_da:", a.grad)
    print("Analytical dl_db:", b.grad)

    # Numerical gradient for c
    numerical_dl_dc = torch.zeros_like(c)
    for i in range(len(c)):
        current_eps = torch.zeros_like(c)
        current_eps[i] = eps
        numerical_dl_dc[i] = (compute_output(c + current_eps, a, b) - compute_output(c - current_eps, a, b)) / (2 * eps)

    print("Numerical dl_dc:", numerical_dl_dc)

    # Numerical gradient for a
    numerical_dl_da = (compute_output(c, a + eps, b) - compute_output(c, a - eps, b)) / (2 * eps)
    print("Numerical dl_da:", numerical_dl_da)

    # Numerical gradient for b

    numerical_dl_db = (compute_output(c, a, b + eps) - compute_output(c, a, b - eps)) / (2 * eps)
    print("Numerical dl_db:", numerical_dl_db)

    # Backward pass



# test_planar_flow()
# run_density_matching_example()
# test_alpha_solve()
run_generation_example('plot','cpu')