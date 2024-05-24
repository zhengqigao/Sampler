import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
import sys
sys.path.append('../')
from sampler._common import BiProbTrans, Distribution
from sampler.model.made import MADE
from test_common_helper import Feedforward, TensorizedMultiGauss, PotentialFunc
import matplotlib.pyplot as plt
from sampler.functional import KLDenLoss, KLGenLoss
import numpy as np
from sklearn import datasets
import numpy as np

# set random seed
# torch.manual_seed(0)

dim = 5
multiple = 1
order = np.random.permutation(range(dim))
model = MADE(hidden_dims=[dim, 20, multiple * dim], order=order)

bs = 1
x = torch.rand(bs, dim)
res = model(x)

ind = 1
x[0, ind] += 10  # Correctly modify the element in x
res2 = model(x)
diff = torch.abs(res - res2)

# Find indices where diff is not zero
indices = diff==0
print(f"Calculated: Final NN output doesn't change indices = {torch.arange(multiple * dim).view(-1,multiple * dim)[indices]}")
print(f"order = {order}")
print(f"change ind={ind}")
print(f"Golden: the following indices shouldn't change (repeat {multiple} times, w/ period {dim}): {torch.arange(dim)[order<=order[ind]]}")

print(model.mask)
print(model.mask_matrix)