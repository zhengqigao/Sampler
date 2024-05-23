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
                 order: Optional[Union[Tuple, List, torch.Tensor]] = None):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.gaussian = gaussian
        self.order = order

        self.hidden_layers = nn.ModuleList(
            [MaskedLinear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)])

        self.mask = [None for _ in range(len(hidden_dims))]
        self.mask_matrix = []
        self.set_binary_mask()

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value: Optional[Union[Tuple, List, torch.Tensor]] = None):
        if value is not None:
            order = torch.Tensor(value)
            if order.min().item() != 0 or order.max().item() != self.hidden_dims[0] - 1 or len(torch.unique(order)) != self.hidden_dims[0]:
                raise ValueError("The order should be a permutation of 0, 1, ..., hidden_dim[0]-1.")
        else:
            order = torch.arange(self.hidden_dims[0])
        self._order = order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.hidden_layers) - 1):
            x = self.activation(self.hidden_layers[i](x))
        x = self.hidden_layers[-1](x)
        return x

    def set_binary_mask(self) -> None:
        L, D = len(self.hidden_dims), self.hidden_dims[0]
        self.mask[0] = torch.Tensor(self.order) if self.order is not None else torch.arange(D)

        for l in range(1, L - 1):
            low = self.mask[l-1].min().item()
            size = self.hidden_dims[l]
            self.mask[l] = torch.randint(low=low, high=D - 1, size=(size,))
        self.mask[L - 1] = self.mask[0] # the output layer is the same as the input layer

        for l in range(L - 1):
            if l == L - 2:
                mask_matrix = self.mask[l + 1].unsqueeze(1) > self.mask[l].unsqueeze(0)
            else:
                mask_matrix = self.mask[l + 1].unsqueeze(1) >= self.mask[l].unsqueeze(0)
            self.mask_matrix.append(mask_matrix)
            self.hidden_layers[l].set_binary_mask(mask_matrix)