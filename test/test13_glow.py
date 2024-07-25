import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
import sys
sys.path.append('../')
from sampler._common import BiProbTrans, Distribution
from sampler.model.net import ConvNet2d
from sampler.model.glow import Actnorm, Inv1by1Conv, Glowblock, MultiscaleFlow, Split, Squeeze
from test_common_helper import Feedforward, TensorizedMultiGauss, PotentialFunc, ClassCondGauss
import matplotlib.pyplot as plt
from sampler.functional import KLDenLoss, KLGenLoss
import numpy as np
from sklearn import datasets
import torchvision as tv
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def test_actnorm():
    shape = (3, 4, 4)
    mg = TensorizedMultiGauss(mean=torch.zeros(*shape), std=torch.ones(*shape))
    flowtransform = Actnorm(num_features=shape[0], p_base=mg)
    print(flowtransform.p_base)
    x = torch.rand(100, *shape)
    x_, log_prob = flowtransform.backward(*flowtransform.forward(x, 0))
    print(log_prob.shape)
    #print(f"diff = {torch.max(torch.abs(x - x_)):.3e}, log_prob = {log_prob.abs().max():.3e}")

def test_1by1conv():
    shape = (10, 4, 4) #c,h,w??
    mg = TensorizedMultiGauss(mean=torch.zeros(*shape), std=torch.ones(*shape))
    flowtransform = Inv1by1Conv(num_features=shape[0], p_base=mg, bias=True)
    print(flowtransform.p_base)
    x = torch.rand(100, *shape)
    x_, log_prob = flowtransform.backward(*flowtransform.forward(x, 0))
    print(log_prob)
    #print(f"diff = {torch.max(torch.abs(x - x_)):.3e}, log_prob = {log_prob.abs().max():.3e}")

def test_Glowblock():
    scalenet = ConvNet2d(
        channels=[3, 12, 12, 6],
        kernel_size=[3, 1, 3],
    )
    shape = (6, 4, 4)
    x = torch.rand(100, *shape)
    test_block = Glowblock(num_features=6,
                           num_trans=3,
                           scale_net=scalenet)
    x_, log_prob = test_block.forward(*test_block.backward(x, 0))
    print(log_prob)
    #print(f"diff = {torch.max(torch.abs(x - x_)):.3e}, log_prob = {log_prob.abs().max():.3e}")

def test_MultiscaleFlow():

    # Define flows
    L = 3
    K = 3
    torch.manual_seed(10)
    input_shape = (3, 8, 8)
    num_elements = np.prod(input_shape)
    channels = 3
    hidden_channels = 12
    split_mode = 'pos'
    scale = True
    num_classes = 4

    # Set up flows, distributions and merge operations
    p_base_list = []
    splits = []
    flows = []
    for i in range(L):
        flows_ = []
        flows_ += [Squeeze()]
        for j in range(K):
            scalenet = ConvNet2d(
                channels=[channels * 2 ** (i+1), hidden_channels, hidden_channels, channels * 2 ** (i+2)],
                kernel_size=[3, 1, 3],
            )
            flows_ += [Glowblock(num_features=channels * 2 ** (i+2),
                                 num_trans=channels * 2 ** (i+1),
                                 scale_net=scalenet)
                       ]  # the number of channels doubles after squeeze and split
        flows += [flows_]
        if i < L-1:
            splits += [Split()]
            latent_shape = (input_shape[0] * 2 ** (i+1), input_shape[1] // 2 ** (i+1),
                            input_shape[2] // 2 ** (i+1))
        else:
            latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L,
                            input_shape[2] // 2 ** L)
        p_base_list += [ClassCondGauss(latent_shape, num_classes)]

    # Construct flow model with the multiscale architecture
    model = MultiscaleFlow(p_base_list, flows, splits)

    # Move model on GPU if available
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    model = model.to(device)

    x = torch.rand(100, *input_shape)
    x_, log_prob = model.forward(*model.backward(x, 0))
    #logp = model.log_prob(x, torch.tensor([0]*10))
    #print(logp)
    #print(f"diff = {torch.max(torch.abs(x - x_)):.3e}, log_prob = {log_prob.abs().max():.3e}")
    # Train model
    max_iter = 20000

    loss_hist = np.array([])

    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for i in range(10):
        x_, y = x[i*10:(i+1)*10], torch.tensor([0]*5+[1]*5)
        optimizer.zero_grad()
        loss = -torch.mean(model.log_prob(x_.to(device), y.to(device)))
        loss.backward()
        optimizer.step()
        loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
    #print(model.p_base[0].shift)
    print(model.sample(10, torch.tensor([0]*5+[1]*5))[0].shape)
    plt.figure(figsize=(10, 10))
    plt.plot(loss_hist, label='loss')
    plt.legend()
    plt.show()

def test_MultiscaleFlow_image():


    batch_size = 256

    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    #tv.transforms.Normalize((0.1307,), (0.3081,))
    train_data = tv.datasets.MNIST('/home/chenzihao/nlguo_tmp/datasets', train=True,
                                    download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                            drop_last=True)

    test_data = tv.datasets.MNIST('/home/chenzihao/nlguo_tmp/datasets', train=False,
                                    download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    train_iter = iter(train_loader)

        # Define flows
    L = 2
    K = 4
    torch.manual_seed(10)
    input_shape = (1, 28, 28)
    num_elements = np.prod(input_shape)
    channels = 1
    hidden_channels = 16
    split_mode = 'pos'
    scale = True
    num_classes = 10

    # Set up flows, distributions and merge operations
    p_base_list = []
    splits = []
    flows = []
    for i in range(L):
        flows_ = []
        flows_ += [Squeeze()]
        for j in range(K):
            scalenet = ConvNet2d(
                channels=[channels * 2 ** (i+1), hidden_channels, hidden_channels, channels * 2 ** (i+2)],
                kernel_size=[3, 1, 3],
            )
            flows_ += [Glowblock(num_features=channels * 2 ** (i+2),
                                 num_trans=channels * 2 ** (i+1),
                                 scale_net=scalenet,
                                 mode=(j%2==0))
                       ]  # the number of channels doubles after squeeze and split
        flows += [flows_]
        if i < L-1:
            splits += [Split()]
            latent_shape = (input_shape[0] * 2 ** (i+1), input_shape[1] // 2 ** (i+1),
                            input_shape[2] // 2 ** (i+1))
        else:
            latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L,
                            input_shape[2] // 2 ** L)
        p_base_list += [ClassCondGauss(latent_shape, num_classes)]

    # Construct flow model with the multiscale architecture
    model = MultiscaleFlow(p_base_list, flows, splits)

    # Move model on GPU if available
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    model = model.to(device)


    max_iter = 50000

    loss_hist = np.array([])

    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for i in tqdm(range(max_iter)):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        optimizer.zero_grad()
        loss = -torch.mean(model.log_prob(x.to(device), y.to(device)))
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
        del(x, y, loss)

        if i % 1000 == 0:
            num_sample = 10

            with torch.no_grad():
                y = torch.arange(num_classes).repeat(num_sample).to(device)
                x, _ = model.sample(y=y)
                x_ = torch.clamp(x, 0, 1)
                plt.figure(figsize=(10, 10))
                plt.imshow(np.transpose(tv.utils.make_grid(x_, nrow=num_classes).cpu().numpy()*255, (1, 2, 0)),cmap="grey")
                plt.savefig(f"current_result_glowh4{i}.png")

    plt.figure(figsize=(10, 10))
    plt.plot(loss_hist, label='loss')
    plt.legend()
    plt.savefig("loss.png")
    
if __name__ == '__main__':
    #test_actnorm()
    #test_1by1conv()
    #test_Glowblock()
    #test_MultiscaleFlow()
    test_MultiscaleFlow_image()