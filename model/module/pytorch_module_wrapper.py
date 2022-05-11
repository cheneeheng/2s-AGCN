from collections import OrderedDict
from typing import Optional, Union, Type

import inspect

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module as PyTorchModule
from torch.nn import Sequential


__all__ = ['Module', 'Conv1xN', 'Conv', 'Pool', 'ASPP']


class Module(PyTorchModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 dilation: Union[int, list] = 1,
                 bias: int = 0,
                 deterministic: Optional[bool] = None,
                 dropout: Optional[Type[PyTorchModule]] = None,
                 activation: Optional[Type[PyTorchModule]] = None,
                 normalization: Optional[Type[PyTorchModule]] = None,
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
                 dropout: Optional[Type[PyTorchModule]] = None,
                 activation: Optional[Type[PyTorchModule]] = None,
                 normalization: Optional[Type[PyTorchModule]] = None,
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
                 dropout: Optional[Type[PyTorchModule]] = None,
                 activation: Optional[Type[PyTorchModule]] = None,
                 normalization: Optional[Type[PyTorchModule]] = None,
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


class ASPP(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dilation: list = [1, 3, 5, 7],
                 bias: int = 0,
                 dropout: Type[PyTorchModule] = nn.Dropout2d,
                 activation: Type[PyTorchModule] = nn.ReLU,
                 normalization: Type[PyTorchModule] = nn.BatchNorm2d):
        super(ASPP, self).__init__(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   dilation=dilation,
                                   bias=bias,
                                   dropout=dropout,
                                   activation=activation,
                                   normalization=normalization)
        self.aspp = torch.nn.ModuleDict()
        self.pool = 0 in self.dilation
        if self.pool:
            self.aspp.update({
                'aspp_pool': Pool(self.in_channels,
                                  self.out_channels,
                                  bias=self.bias,
                                  pooling=nn.AdaptiveAvgPool2d(1),
                                  activation=self.activation)
            })
        for dil in self.dilation:
            if dil == 0:
                continue
            self.aspp.update({
                f'aspp_{dil}': Conv(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    padding=dil,
                    dilation=dil,
                    bias=self.bias,
                    activation=self.activation,
                    normalization=lambda: self.normalization(
                        self.out_channels),
                    deterministic=False
                )
            })
        if len(inspect.getargspec(self.dropout).args) != 0:
            def dropout(): return self.dropout(0.2)
        else:
            dropout = self.dropout
        self.proj = Conv(
            self.out_channels * len(self.dilation),
            self.out_channels,
            bias=self.bias,
            normalization=lambda: self.normalization(self.out_channels),
            dropout=dropout
        )
        if self.in_channels == self.out_channels:
            self.res = nn.Identity()
        else:
            self.res = Conv(self.in_channels,
                            self.out_channels,
                            bias=self.bias)

    def forward(self,
                x: torch.Tensor,
                x_n: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        # x: n,c,v,t
        res = []
        for _, block in self.aspp.items():
            res.append(block(x))
        if self.pool:
            res[0] = F.interpolate(res[0],
                                   size=x.shape[-2:],
                                   mode="bilinear",
                                   align_corners=False)
        res = torch.cat(res, dim=1)
        x = self.proj(res) + self.res(x)
        return x
