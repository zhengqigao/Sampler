import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
import sys
sys.path.append('../')
from sampler._common import BiProbTrans, Distribution
from sampler.model.planarflow import PlanarFlow
from test_common_helper import Feedforward, MultiGauss, PotentialFunc


def test_planar_flow():
    dim = 4
    mg = MultiGauss(mean=[0] * dim, std=[1] * dim)
    flowtransform = PlanarFlow(dim=dim, num_trans=3, p_base= mg)
    print(flowtransform.p_base)
    x = torch.rand(10, dim)
    x._requires_grad = True
    z, log_prob = flowtransform(x)
    x_, log_prob_ = flowtransform.backward(z)
    diff = x - x_
    print(f"diff = {torch.max(torch.abs(diff))}")


test_planar_flow()