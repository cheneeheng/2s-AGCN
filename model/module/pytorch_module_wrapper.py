from collections import OrderedDict
from typing import Optional, Union, Type

import inspect


try:
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
except ImportError:
    print("Warning: fvcore is not found")

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module as PyTorchModule
from torch.nn import Sequential


__all__ = ['Module', 'Conv1xN', 'Conv', 'Pool', 'ASPP']

OTPM = Optional[Type[PyTorchModule]]


class Module(PyTorchModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 dilation: Union[int, list] = 1,
                 bias: int = 0,
                 deterministic: Optional[bool] = None,
                 dropout: OTPM = None,
                 activation: OTPM = None,
                 normalization: OTPM = None,
                 prenorm: bool = False
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
        self.prenorm = prenorm

    def update_block_dict(self, block: OrderedDict) -> OrderedDict:
        if self.normalization is not None:
            block.update({'norm': self.normalization()})
            if self.prenorm:
                block.move_to_end('norm', last=False)
        if self.activation is not None:
            block.update({'act': self.activation()})
        if self.dropout is not None:
            block.update({'dropout': self.dropout()})
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv1xN(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: int = 0,
                 deterministic: Optional[bool] = None,
                 ):
        super(Conv1xN, self).__init__(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      dilation=dilation,
                                      bias=bias,
                                      deterministic=deterministic)
        assert isinstance(self.kernel_size, int)
        assert isinstance(self.padding, int)
        assert isinstance(self.dilation, int)
        self.conv = nn.Conv2d(self.in_channels,
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
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: int = 0,
                 deterministic: Optional[bool] = None,
                 dropout: OTPM = None,
                 activation: OTPM = None,
                 normalization: OTPM = None,
                 prenorm: bool = False
                 ):
        super(Conv, self).__init__(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   dilation=dilation,
                                   bias=bias,
                                   deterministic=deterministic,
                                   dropout=dropout,
                                   activation=activation,
                                   normalization=normalization,
                                   prenorm=prenorm)
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
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pooling: PyTorchModule,
                 kernel_size: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: int = 0,
                 deterministic: Optional[bool] = None,
                 dropout: OTPM = None,
                 activation: OTPM = None,
                 normalization: OTPM = None,
                 prenorm: bool = False
                 ):
        super(Pool, self).__init__(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   dilation=dilation,
                                   bias=bias,
                                   deterministic=deterministic,
                                   dropout=dropout,
                                   activation=activation,
                                   normalization=normalization,
                                   prenorm=prenorm)
        block = OrderedDict({
            'pool': pooling,
            'conv': Conv1xN(self.in_channels,
                            self.out_channels,
                            bias=self.bias),
        })
        block = self.update_block_dict(block)
        self.block = Sequential(block)


class ASPP(PyTorchModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 #  padding: Optional[list] = None,
                 dilation: list = [1, 3, 5, 7],
                 bias: int = 0,
                 dropout: Type[PyTorchModule] = None,  # nn.Dropout2d,
                 activation: Type[PyTorchModule] = None,  # nn.ReLU,
                 normalization: Type[PyTorchModule] = None,  # nn.BatchNorm2d,
                 residual: int = 0):
        super(ASPP, self).__init__()

        _dropout = dropout
        if len(inspect.getargspec(dropout).args) != 0:
            def _dropout(): return dropout(0.2)

        _normalization = normalization
        if len(inspect.getargspec(normalization).args) != 0:
            def _normalization(): return normalization(out_channels)

        self.block = torch.nn.ModuleDict()

        for i in range(len(dilation)):
            if dilation[i] == 0:
                self.block.update({
                    'aspp_pool': Pool(in_channels,
                                      out_channels,
                                      bias=bias,
                                      pooling=nn.AdaptiveAvgPool2d(1),
                                      activation=activation)
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
                         activation=activation,
                         normalization=_normalization,
                         deterministic=False)
                })

        self.proj = Conv(out_channels * len(dilation),
                         out_channels,
                         bias=bias,
                         normalization=_normalization,
                         dropout=_dropout)

        if residual == 0:
            self.res = lambda x: 0
        elif in_channels == out_channels:
            self.res = nn.Identity()
        else:
            self.res = Conv(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: n,c,v,t
        res = []
        for name, block in self.block.items():
            x1 = block(x)
            if name == 'aspp_pool':
                x1 = F.interpolate(x1,
                                   size=x.shape[-2:],
                                   mode="bilinear",
                                   align_corners=False)
            res.append(x1)
        res = torch.cat(res, dim=1)
        x = self.proj(res) + self.res(x)
        return x


if __name__ == '__main__':
    inputs = torch.ones(1, 3, 12, 12)
    model = ASPP(3, 10)
    flops = FlopCountAnalysis(model, inputs)
    print(flops.total())
    # print(flops.by_module_and_operator())
