import torch.nn as nn
from typing import List

class ConvNet2d(nn.Module):
    r"""
    Convolutional Neural Networks, actnorm is not considered yet
    """

    def __init__(
            self,
            channels: List[int],
            kernel_size: List[int],
            leaky: float = 0.0,
            init_zeros: bool = True,
            weight_std: float = None,
    ):
        super().__init__()
        net = nn.ModuleList([])
        layer_num = len(kernel_size)
        for i in range(layer_num - 1):
            conv = nn.Conv2d(
                channels[i],
                channels[i + 1],
                kernel_size[i],
                padding="same",
                bias=True,
            )
            if weight_std is not None:
                conv.weight.data.normal_(mean=0.0, std=weight_std)
            net.append(conv)
            net.append(nn.LeakyReLU(leaky))
        net.append(
            nn.Conv2d(
                channels[layer_num - 1],
                channels[layer_num],
                kernel_size[layer_num - 1],
                padding="same",
            )
        )
        if init_zeros:
            nn.init.zeros_(net[-1].weight) #TODO: check why only the last layer is initialized with zero
            nn.init.zeros_(net[-1].bias)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
