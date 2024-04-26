import sys
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../"))
import numpy as np
from sampler.model import CouplingFlow, RealNVP
from test_common_helper import Feedforward, MultiGauss, PotentialFunc
from sampler._common import Distribution
from sampler.base import importance_sampling
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn import datasets
from sampler.functional import KLDenLoss, KLGenLoss, ScoreDenLoss
# test a single transform block

def test_couple_flow():
    dim = 4
    mg = MultiGauss(mean=[0] * dim, std=[1] * dim)
    flowtransform = CouplingFlow(dim=dim,
                               keep_dim=[0, 2],
                               scale_net=Feedforward([max(1, dim // 2), 2, 2, max(1, dim // 2)], 'relu'),
                               shift_net=Feedforward([max(1, dim // 2), 2, 2, max(1, dim // 2)], 'relu'),
                               p_base=mg)

    print(flowtransform.p_base)
    flowtransform.p_base = None
    print(flowtransform.p_base)
    flowtransform.p_base = mg
    print(flowtransform.p_base)

    x = torch.rand(10, dim)
    x_, diff_log_det = flowtransform.backward(*flowtransform.forward(x, 0))
    diff = x - x_
    print(f"diff = {torch.max(torch.abs(diff))}")
    print(f"diff_log_det = {torch.max(torch.abs(diff_log_det))}")

    samples, log_prob = flowtransform.sample(10)
    flowtransform.p_base = None
    try:
        samples, log_prob = flowtransform.sample(10)
    except Exception as e:
        print("Error raised as expected, w/ error: ", e)
    flowtransform.p_base = mg
    samples, log_prob = flowtransform.sample(10)

    # test if flowtransform is compatible with IS
    flowtransform.p_base = mg
    results, _ = importance_sampling(10000, flowtransform, flowtransform, lambda x: x)

    # after run IS, test again if flowtransform is still working (backward, forward, sample)
    x = torch.rand(10, dim)
    x_, diff_log_det = flowtransform.backward(*flowtransform.forward(x, 0))
    diff = x - x_
    print(f"diff = {torch.max(torch.abs(diff))}")
    print(f"diff_log_det = {torch.max(torch.abs(diff_log_det))}")
    samples, log_prob = flowtransform.sample(10)

    try:
        results, _ = importance_sampling(10000, flowtransform, flowtransform, lambda x: x)
    except Exception as e:
        print("Error raised as expected, w/ error: ", e)
        flowtransform.restore()  # restore because IS is interrupted in the middle, so we need to restore the model to its original status manually. This won't be needed in the real-world use cases. If not restored, then the sample and forward methods have been modified.


dim = 4
mg = MultiGauss(mean=[0] * dim, std=[1] * dim)
# # test a flow model
num_trans = 4
nf = RealNVP(dim=dim,
             num_trans=num_trans,
             scale_net=Feedforward([max(1, dim // 2), 2, 2, max(1, dim // 2)], 'relu'),
             shift_net=nn.ModuleList(
                 [Feedforward([max(1, dim // 2), 2, 2, max(1, dim // 2)], 'relu') for _ in
                  range(num_trans)]),
             keep_dim=[[0, 2], [1, 3], [0, 2], [1, 3]],
             p_base=mg)


def test_backward_forward():
    y = torch.rand(10, dim)
    y_, diff_log_det = nf.backward(*nf.forward(y, 0))
    diff = y - y_
    print(f"diff = {torch.max(torch.abs(diff))}")
    print(f"diff_log_det = {torch.max(torch.abs(diff_log_det))}")


def test_compatiblity_with_IS():
    nf.p_base = mg
    results, _ = importance_sampling(10000, nf, nf, lambda x: x)
    nf.p_base = None
    try:
        results, _ = importance_sampling(10000, nf, nf, lambda x: x)
    except Exception as e:
        print("Error raised as expected, w/ error: ", e)


def test_sample():
    nf = RealNVP(dim=2,
                 num_trans=2,
                 scale_net=None,
                 shift_net=None,
                 p_base=MultiGauss(mean=[0, 0], std=[1, 1]))
    # essentially nf will still be a Gaussian.
    samples, log_prob = nf.sample(10000)
    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1], c=log_prob, cmap='viridis')
    plt.show()

    target = MultiGauss(mean=[-1, 1], std=[1, 1])

    results, _ = importance_sampling(10000, target, nf, lambda x: x)
    print(results)


def run_density_matching_example():
    ## TODO: I tested potential3 and potential6, they work. Can you try testing other cases? Note that because of randomness,
    # may need to run with different random seeds to obtain good results. Also, I haven't tested it on GPU.

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

    num_trans = 12
    ## TODO: to(device), module.sample might have error, because p_base is on cpu, while all parameters of module is on gpu.
    module = RealNVP(dim=2,
                     num_trans=num_trans,
                     scale_net=nn.ModuleList(
                         [Feedforward([1, 128, 128, 128, 1], 'leakyrelu') for _ in
                          range(num_trans)]),
                     shift_net=nn.ModuleList(
                         [Feedforward([1, 128, 128, 128, 1], 'leakyrelu') for _ in
                          range(num_trans)]),
                     p_base=MultiGauss(mean=[0, 0], std=[1, 1])).to(device)

    optimizer = torch.optim.Adam(module.parameters(), lr=0.0001)
    max_iter = 500
    loss_list = []
    batch_size = 1000
    criterion1 = KLDenLoss(log_p = lambda x: -potential_func(x))
    criterion2 = ScoreDenLoss(log_p = lambda x: -potential_func(x))

    for i in range(max_iter):
        loss = criterion1(module, batch_size)
        loss_tmp = criterion2(module, batch_size)
        loss_list.append(loss.item())
        if torch.isnan(loss).any() or i == max_iter - 1:
            plt.figure()
            plt.plot(loss_list)
            plt.show()
            break
        optimizer.zero_grad()
        loss_tmp.backward() # loss.backward() should be used, loss_tmp.backward() uses score estimator, and usually has large variance.
        optimizer.step()
        print(f"iter {i}, loss: {loss.item()}, loss_tmp: {loss_tmp.item()}")

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



def run_generation_example():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_trans = 12
    module = RealNVP(dim=2,
                     num_trans=num_trans,
                     scale_net=nn.ModuleList(
                         [Feedforward([1, 128, 128, 128, 1], 'leakyrelu') for _ in
                          range(num_trans)]),
                     shift_net=nn.ModuleList(
                         [Feedforward([1, 128, 128, 128, 1], 'leakyrelu') for _ in
                          range(num_trans)]),
                     p_base=MultiGauss(mean=[0, 0], std=[1, 1])).to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.0001)
    num_steps = 1000
    criterion = KLGenLoss()
    for i in range(num_steps):
        z, _ = datasets.make_moons(n_samples=1000, noise=0.1)
        z = torch.Tensor(z).to(device)
        # x, log_prob = module.log_prob(z)
        # loss = -torch.mean(log_prob) # KL[p||q] = -logq, log_prob = logq
        loss = criterion(module, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"iter {i}, loss: {loss.item()}")

    # show the generated samples
    samples, log_prob = module.sample(10000)
    samples = samples.cpu().detach().numpy()
    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.title("generated samples")

    plt.figure()
    x, _ = datasets.make_moons(n_samples=1000, noise=0.1)
    plt.scatter(x[:, 0], x[:, 1])
    plt.title("real samples")

    plt.show()


run_density_matching_example()
run_generation_example()