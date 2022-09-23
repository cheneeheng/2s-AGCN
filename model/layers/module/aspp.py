from typing import Optional, Union, Type

import torch
from torch import nn
from torch.nn import functional as F

from model.layers.module.block import PyTorchModule
from model.layers.module.block import Conv
from model.layers.module.block import Pool
from model.layers.module.block import OUTPMPM


class ASPP(PyTorchModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dilation: list = [1, 3, 5, 7],
                 bias: int = 0,
                 dropout: OUTPMPM = None,  # nn.Dropout2d,
                 activation: OUTPMPM = None,  # nn.ReLU,
                 normalization: OUTPMPM = None,  # nn.BatchNorm2d,
                 ):
        super(ASPP, self).__init__()

        if isinstance(dropout, Type[PyTorchModule]):
            def _dropout(): return dropout(0.2)
        elif isinstance(dropout, callable):
            _dropout = dropout
        else:
            raise ValueError("Unknown dropout arg in ASPP")

        if isinstance(activation, Type[PyTorchModule]):
            def _activation(): return activation(0.1)
        elif isinstance(activation, callable):
            _activation = activation
        else:
            raise ValueError("Unknown activation arg in ASPP")

        if isinstance(normalization, Type[PyTorchModule]):
            def _normalization(): return normalization(out_channels)
        elif isinstance(normalization, callable):
            _normalization = normalization
        else:
            raise ValueError("Unknown normalization arg in ASPP")

        self.block = nn.ModuleDict()
        for i in range(len(dilation)):
            if dilation[i] == 0:
                self.block.update({
                    'aspp_pool': Pool(in_channels,
                                      out_channels,
                                      pooling=nn.AdaptiveAvgPool2d(1),
                                      bias=bias,
                                      activation=_activation,
                                      normalization=_normalization)
                })
            else:
                self.block.update({
                    f'aspp_{dilation[i]}':
                    Conv(in_channels,
                         out_channels,
                         kernel_size=kernel_size,
                         padding=dilation[i],
                         dilation=dilation[i],
                         bias=bias,
                         activation=_activation,
                         normalization=_normalization,
                         deterministic=False)
                })

        self.projection = Conv(out_channels * len(dilation),
                               out_channels,
                               bias=bias,
                               normalization=_normalization,
                               dropout=_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: n,c,v,t
        output = []
        for name, block in self.block.items():
            z = block(x)
            if name == 'aspp_pool':
                z = F.interpolate(z,
                                  size=x.shape[-2:],
                                  mode="bilinear",
                                  align_corners=False)
            output.append(z)
        x = self.projection(torch.cat(output, dim=1))
        return x
