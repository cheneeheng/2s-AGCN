from collections import OrderedDict
from typing import Tuple, Optional, Union, Type

import torch
from torch.nn import Conv2d
from torch.nn import Module as PyTorchModule
from torch.nn import Sequential

__all__ = ['Module', 'Conv1xN', 'Conv', 'Pool']


class Module(PyTorchModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: int = 0,
                 deterministic: Optional[bool] = None,
                 dropout: Optional[Type[PyTorchModule]] = None,
                 activation: Optional[Type[PyTorchModule]] = None,
                 normalization: Optional[Type[PyTorchModule]] = None,
                 ):
        super(Module, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        if deterministic is None:
            self.deterministic = torch.backends.cudnn.deterministic
        else:
            self.deterministic = deterministic
        self.dropout = dropout
        self.activation = activation
        self.normalization = normalization

    def update_block_dict(self, block: OrderedDict) -> OrderedDict:
        if self.normalization is not None:
            block.update({'norm': self.normalization()})
        if self.activation is not None:
            block.update({'act': self.activation()})
        if self.dropout is not None:
            block.update({'dropout': self.dropout()})
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv1xN(Module):
    def __init__(self, *args, **kwargs):
        super(Conv1xN, self).__init__(*args, **kwargs)
        assert isinstance(self.kernel_size, int)
        assert isinstance(self.padding, int)
        assert isinstance(self.dilation, int)
        self.conv = Conv2d(self.in_channels,
                           self.out_channels,
                           kernel_size=(1, self.kernel_size),
                           padding=(0, self.padding),
                           dilation=self.dilation,
                           bias=bool(self.bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.deterministic:
            torch.backends.cudnn.deterministic = False
            # torch.backends.cudnn.benchmark = True
        x = self.conv(x)
        if not self.deterministic:
            torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = True
        return x


class Conv(Module):
    def __init__(self, *args, **kwargs):
        super(Conv, self).__init__(*args, **kwargs)
        block = OrderedDict({
            'conv': Conv1xN(self.in_channels,
                            self.out_channels,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            dilation=self.dilation,
                            bias=self.bias,
                            deterministic=self.deterministic),
        })
        block = self.update_block_dict(block)
        self.block = Sequential(block)


class Pool(Module):
    def __init__(self, *args, pooling: PyTorchModule, **kwargs):
        super(Pool, self).__init__(*args, **kwargs)
        block = OrderedDict({
            'pool': pooling,
            'conv': Conv1xN(self.in_channels,
                            self.out_channels,
                            bias=self.bias),
        })
        block = self.update_block_dict(block)
        self.block = Sequential(block)
