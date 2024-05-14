import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional


class MaskedLinear(nn.Linear):
    def set_binary_mask(self, mask: torch.Tensor):
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    r"""
    Masked Autoencoder for Distribution Estimation (MADE). Code is copied and refined based on https://github.com/e-hulten/maf/blob/master/made.py

    """
    def __init__(self, hidden_dims: List,
                 activation: Optional[nn.Module] = nn.ReLU(),
                 gaussian: Optional[bool] = False,
                 random_order: Optional[bool] = False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.gaussian = gaussian
        self.random_order = random_order

        self.hidden_layers = nn.ModuleList(
            [MaskedLinear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)])

        self.set_binary_mask()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.hidden_layers) - 1):
            x = self.activation(self.hidden_layers[i](x))
        x = self.hidden_layers[-1](x)
        return x

    def set_binary_mask(self) -> None:
        for i, layer in enumerate(self.hidden_layers):
            mask = torch.zeros(layer.out_features, layer.in_features)
            mask[:, :layer.out_features // 2] = 1
            layer.set_binary_mask(mask)
