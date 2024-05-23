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

dim = 10
order = range(dim) # np.random.permutation(range(dim))
model = MADE(hidden_dims=[dim, 4, 2, dim])

bs = 1
x = torch.rand(bs, dim)
res = model(x)

ind = 3
x[0, ind] += 10  # Correctly modify the element in x
res2 = model(x)
diff = torch.abs(res - res2)

# Find indices where diff is not zero
indices = diff==0
print(f"Final NN output doesn't change indices = {torch.arange(dim).view(-1,dim)[indices]}")
print(f"order = {order}")
print(f"change index = {ind}")