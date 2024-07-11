import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
import sys
sys.path.append('../')
from sampler._common import BiProbTrans, Distribution
from sampler.model.net import ConvNet2d
from sampler.model.glow import Actnorm, Inv1by1Conv, Glowblock
from test_common_helper import Feedforward, TensorizedMultiGauss, PotentialFunc
import matplotlib.pyplot as plt
from sampler.functional import KLDenLoss, KLGenLoss
import numpy as np
from sklearn import datasets


def test_actnorm():
    shape = (3, 4, 4)
    mg = TensorizedMultiGauss(mean=torch.zeros(*shape), std=torch.ones(*shape))
    flowtransform = Actnorm(num_features=shape[0], p_base=mg)
    print(flowtransform.p_base)
    x = torch.rand(100, *shape)
    x_, log_prob = flowtransform.backward(*flowtransform.forward(x, 0))
    print(f"diff = {torch.max(torch.abs(x - x_)):.3e}, log_prob = {log_prob.abs().max():.3e}")

def test_1by1conv():
    shape = (10, 4, 4) #c,h,w??
    mg = TensorizedMultiGauss(mean=torch.zeros(*shape), std=torch.ones(*shape))
    flowtransform = Inv1by1Conv(num_features=shape[0], p_base=mg, bias=True)
    print(flowtransform.p_base)
    x = torch.rand(100, *shape)
    x_, log_prob = flowtransform.backward(*flowtransform.forward(x, 0))
    print(f"diff = {torch.max(torch.abs(x - x_)):.3e}, log_prob = {log_prob.abs().max():.3e}")

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
    x_, log_prob = test_block.backward(*test_block.forward(x, 0))
    print(f"diff = {torch.max(torch.abs(x - x_)):.3e}, log_prob = {log_prob.abs().max():.3e}")



if __name__ == '__main__':
    # test_actnorm()
    # test_1by1conv()
    test_Glowblock()