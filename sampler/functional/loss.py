import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
from .._common import BiProbTrans, Distribution

def KL(samples: torch.Tensor,
                    model: BiProbTrans,
                    ):
    r"""
    The generation loss for the model.

    Args:
        samples (torch.Tensor): The samples from the model.
        model (BiProbTrans): The model to calculate the generation loss.
    """

