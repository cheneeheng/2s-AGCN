import torch
from torch import nn
from torch import Tensor
from torch.nn import Module as PyTorchModule
from torch.nn import functional as F
from torch import einsum

from einops import rearrange

from typing import Tuple, Optional, Union, Type, List

from model.layers import Module
from model.layers import Conv
from model.layers import residual as res
from model.layers import null_fn
from model.layers import get_activation_fn

from model.architecture.sgn.blocks import OneHotTensor
from model.architecture.sgn.blocks import Embedding
from model.architecture.sgn.blocks import MLPTemporal

T1 = Type[PyTorchModule]
T2 = List[Optional[Type[PyTorchModule]]]
T3 = Union[List[int], int]

# Contains:
# - ffn layer
# - G layer
# - GT layer = G layer + another attention matrix for temporal
# - GCN unit = pure gcn operation only
# - GCN block = gcn block that adds ffn or residual is needed
# - GCN block2 = another gcns for the fpn mode = 10


class GCNSpatialFFN(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: int = 0,
                 activation: T1 = nn.ReLU,
                 normalization: T1 = nn.BatchNorm2d,
                 multiplier: float = 4.0
                 ):
        super(GCNSpatialFFN, self).__init__(in_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            padding=padding,
                                            bias=bias,
                                            activation=activation,
                                            normalization=normalization)
        self.ffn1 = Conv(self.in_channels,
                         int(self.in_channels*multiplier),
                         bias=self.bias,
                         kernel_size=self.kernel_size,
                         padding=self.padding,
                         activation=self.activation,
                         normalization=lambda: self.normalization(
                             int(self.in_channels*multiplier)))
        self.ffn2 = Conv(int(self.in_channels*multiplier),
                         self.out_channels,
                         bias=self.bias,
                         kernel_size=self.kernel_size,
                         padding=self.padding,
                         activation=self.activation,
                         normalization=lambda: self.normalization(
                             self.out_channels))

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.ffn1(x)
        x1 = self.ffn2(x1)
        x1 = x1 + x
        return x1


class GCNSpatialG(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: int = 0,
                 activation: T1 = nn.Softmax,
                 g_proj_shared: bool = False,
                 **kwargs):
        super(GCNSpatialG, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=bias,
                                          activation=activation)
        if self.kernel_size == 0:
            self.return_none = True
        else:
            self.return_none = False
            self.g1 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                           kernel_size=self.kernel_size, padding=self.padding)
            if g_proj_shared:
                self.g2 = self.g1
            else:
                self.g2 = Conv(self.in_channels, self.out_channels,
                               bias=self.bias, kernel_size=self.kernel_size,
                               padding=self.padding)
            try:
                self.act = self.activation(dim=-1)
            except TypeError:
                self.act = self.activation()
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, g: Optional[Tensor] = None) -> Tensor:
        if self.return_none:
            return None
        else:
            g1 = self.g1(x).permute(0, 3, 2, 1).contiguous()  # n,t,v,c
            g2 = self.g2(x).permute(0, 3, 1, 2).contiguous()  # n,t,c,v
            g3 = g1.matmul(g2)  # n,t,v,v
            g4 = self.act(g3)
            if g is not None:
                g4 = (g * self.alpha + g4) / (self.alpha + 1)
            return g4, None


class GCNSpatialGT(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: int = 0,
                 activation: T1 = nn.Softmax,
                 g_proj_shared: bool = False,
                 gt_activation: int = 1,
                 num_segment: int = 20,
                 **kwargs
                 ):
        super(GCNSpatialGT, self).__init__(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           padding=padding,
                                           bias=bias,
                                           activation=activation)
        if self.kernel_size == 0:
            self.return_none = True

        else:
            self.return_none = False
            self.g1 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                           kernel_size=self.kernel_size, padding=self.padding)
            self.g3 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                           kernel_size=self.kernel_size, padding=self.padding)
            if g_proj_shared:
                self.g2 = self.g1
                self.g4 = self.g3
            else:
                self.g2 = Conv(self.in_channels, self.out_channels,
                               bias=self.bias, kernel_size=self.kernel_size,
                               padding=self.padding)
                self.g4 = Conv(self.in_channels, self.out_channels,
                               bias=self.bias, kernel_size=self.kernel_size,
                               padding=self.padding)
            self.g3p = nn.AdaptiveMaxPool2d((1, num_segment))
            self.g4p = nn.AdaptiveMaxPool2d((1, num_segment))
            try:
                self.act1 = self.activation(dim=-1)
            except TypeError:
                self.act1 = self.activation()
            if gt_activation == 1:
                self.act2 = self.act1
            elif gt_activation == 2:
                self.act2 = nn.Sigmoid()
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, g: Optional[Tensor] = None) -> Tuple[Tensor]:
        if self.return_none:
            return None, None

        else:
            _, _, v, _ = x.shape

            g1 = self.g1(x).permute(0, 3, 2, 1).contiguous()  # n,t,v,c
            g2 = self.g2(x).permute(0, 3, 1, 2).contiguous()  # n,t,c,v
            g12 = g1.matmul(g2)  # n,t,v,v

            g3 = self.g3(x)  # n,c,v,t
            g4 = self.g4(x)  # n,c,v,t
            g3p = self.g3p(g3)  # n,c,1,t
            g4p = self.g4p(g4)  # n,c,1,t

            g3p = rearrange(g3p, 'n c v t -> n t (c v)')  # n,t,c
            g4p = rearrange(g4p, 'n c v t -> n (c v) t')  # n,c,t
            g34 = einsum('n i d, n d j -> n i j', g3p, g4p)  # n,t,t
            g34 = self.act1(g34)  # n,t,t'

            g12 = rearrange(g12, 'n t i j -> n t (i j)')  # n,t,vv
            g12 = g34.matmul(g12)  # n,t,vv
            g12 = rearrange(g12, 'n t (i j) -> n t i j', i=v)  # n,t,v,v
            g12 = self.act2(g12)  # n,t,v,v'

            if g is not None:
                g12 = (g * self.alpha + g12) / (self.alpha + 1)

            return g12, g34


class GCNSpatialGT2(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: int = 0,
                 activation: T1 = nn.Softmax,
                 g_proj_shared: bool = False,
                 gt_activation: int = 1,
                 num_joint: int = 25,
                 **kwargs
                 ):
        super(GCNSpatialGT2, self).__init__(in_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            padding=padding,
                                            bias=bias,
                                            activation=activation)
        if self.kernel_size == 0:
            self.return_none = True

        else:
            self.return_none = False
            self.g1 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                           kernel_size=self.kernel_size, padding=self.padding)
            if g_proj_shared:
                self.g2 = self.g1
            else:
                self.g2 = Conv(self.in_channels, self.out_channels,
                               bias=self.bias, kernel_size=self.kernel_size,
                               padding=self.padding)
            self.g3 = nn.Linear(self.in_channels*num_joint, 1, bias=self.bias)
            try:
                self.act1 = self.activation(dim=-1)
            except TypeError:
                self.act1 = self.activation()
            if gt_activation == 1:
                self.act2 = self.act1
            elif gt_activation == 2:
                self.act2 = nn.Sigmoid()
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, g: Optional[Tensor] = None) -> Tuple[Tensor]:
        if self.return_none:
            return None, None

        else:
            g1 = self.g1(x).permute(0, 3, 2, 1).contiguous()  # n,t,v,c
            g2 = self.g2(x).permute(0, 3, 1, 2).contiguous()  # n,t,c,v
            g12 = g1.matmul(g2)  # n,t,v,v
            g12 = self.act1(g12)  # n,t,v,v'

            x3 = rearrange(x, 'n c v t -> n t (c v)')  # n,t,cv
            g3 = self.g3(x3).squeeze(-1)  # n,t
            g3 = self.act2(g3)  # n,t'
            g3 = g3.unsqueeze(-1).unsqueeze(-1)  # n,t',1,1

            g12 = g3*g12

            if g is not None:
                g12 = (g * self.alpha + g12) / (self.alpha + 1)

            return g12, g3


class GCNSpatialGT3(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: int = 0,
                 activation: T1 = nn.Softmax,
                 g_proj_shared: bool = False,
                 gt_activation: int = 1,
                 num_joint: int = 25,
                 kernel_size2: int = 3,
                 **kwargs
                 ):
        super(GCNSpatialGT3, self).__init__(in_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            padding=padding,
                                            bias=bias,
                                            activation=activation)
        if self.kernel_size == 0:
            self.return_none = True

        else:
            self.return_none = False
            self.g1 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                           kernel_size=self.kernel_size, padding=self.padding)
            if g_proj_shared:
                self.g2 = self.g1
            else:
                self.g2 = Conv(self.in_channels, self.out_channels,
                               bias=self.bias, kernel_size=self.kernel_size,
                               padding=self.padding)

            idx = 2
            self.g3 = MLPTemporal(
                channels=[self.in_channels*num_joint] * idx + [1],
                kernel_sizes=[kernel_size2] * (idx-1) + [1],
                paddings=[kernel_size2//2] * (idx-1) + [0],
                dilations=[1] * idx,
                biases=[self.bias] * idx,
                residuals=[0] * idx,
                dropouts=[nn.Dropout2d] + [None] * (idx-1),
                activations=[nn.ReLU] * (idx-1) + [None],
                normalizations=[nn.BatchNorm2d] * (idx-1) + [None],
            )
            try:
                self.act1 = self.activation(dim=-1)
            except TypeError:
                self.act1 = self.activation()
            if gt_activation == 1:
                self.act2 = self.act1
            elif gt_activation == 2:
                self.act2 = nn.Sigmoid()
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, g: Optional[Tensor] = None) -> Tuple[Tensor]:
        if self.return_none:
            return None, None

        else:
            g1 = self.g1(x).permute(0, 3, 2, 1).contiguous()  # n,t,v,c
            g2 = self.g2(x).permute(0, 3, 1, 2).contiguous()  # n,t,c,v
            g12 = g1.matmul(g2)  # n,t,v,v
            g12 = self.act1(g12)  # n,t,v,v'

            x3 = rearrange(x, 'n c v t -> n (c v) t').unsqueeze(2)  # n,cv,1,t
            g3 = self.g3(x3).squeeze(1).squeeze(1)  # n,t
            g3 = self.act2(g3)  # n,t'
            g3 = g3.unsqueeze(-1).unsqueeze(-1)  # n,t',1,1

            g12 = g3*g12

            if g is not None:
                g12 = (g * self.alpha + g12) / (self.alpha + 1)

            return g12, g3


class GCNSpatialGT4(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: int = 0,
                 activation: T1 = nn.Softmax,
                 g_proj_shared: bool = False,
                 gt_activation: int = 1,
                 num_joint: int = 25,
                 kernel_size2: int = 3,
                 g3_idx: int = 2,
                 **kwargs
                 ):
        super(GCNSpatialGT4, self).__init__(in_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            padding=padding,
                                            bias=bias,
                                            activation=activation)
        if self.kernel_size == 0:
            self.return_none = True

        else:
            self.return_none = False
            self.g1 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                           kernel_size=self.kernel_size, padding=self.padding)
            if g_proj_shared:
                self.g2 = self.g1
            else:
                self.g2 = Conv(self.in_channels, self.out_channels,
                               bias=self.bias, kernel_size=self.kernel_size,
                               padding=self.padding)

            idx = g3_idx
            assert idx > 1
            self.g3 = MLPTemporal(
                channels=[self.in_channels*num_joint] +
                [self.in_channels] * (idx-1) + [1],
                kernel_sizes=[kernel_size2] * (idx-1) + [1],
                paddings=[kernel_size2//2] * (idx-1) + [0],
                dilations=[1] * idx,
                biases=[self.bias] * idx,
                residuals=[0] * idx,
                dropouts=[nn.Dropout2d] + [None] * (idx-1),
                activations=[nn.ReLU] * (idx-1) + [None],
                normalizations=[nn.BatchNorm2d] * (idx-1) + [None],
            )
            try:
                self.act1 = self.activation(dim=-1)
            except TypeError:
                self.act1 = self.activation()
            if gt_activation == 1:
                self.act2 = self.act1
            elif gt_activation == 2:
                self.act2 = nn.Sigmoid()
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, g: Optional[Tensor] = None) -> Tuple[Tensor]:
        if self.return_none:
            return None, None

        else:
            g1 = self.g1(x).permute(0, 3, 2, 1).contiguous()  # n,t,v,c
            g2 = self.g2(x).permute(0, 3, 1, 2).contiguous()  # n,t,c,v
            g12 = g1.matmul(g2)  # n,t,v,v
            g12 = self.act1(g12)  # n,t,v,v'

            x3 = rearrange(x, 'n c v t -> n (c v) t').unsqueeze(2)  # n,cv,1,t
            g3 = self.g3(x3).squeeze(1).squeeze(1)  # n,t
            g3 = self.act2(g3)  # n,t'
            g3 = g3.unsqueeze(1).unsqueeze(1)  # n,t',1,1

            return g12, g3


class GCNSpatialGT5(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: int = 0,
                 activation: T1 = nn.Softmax,
                 g_proj_shared: bool = False,
                 num_joint: int = 25,
                 num_segment: int = 20,
                 out_channels2: int = 512,
                 kernel_size2: int = 3,
                 g3_idx: int = 2,
                 **kwargs
                 ):
        super(GCNSpatialGT5, self).__init__(in_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            padding=padding,
                                            bias=bias,
                                            activation=activation)
        if self.kernel_size == 0:
            self.return_none = True

        else:
            self.return_none = False
            self.g1 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                           kernel_size=self.kernel_size, padding=self.padding)
            if g_proj_shared:
                self.g2 = self.g1
            else:
                self.g2 = Conv(self.in_channels, self.out_channels,
                               bias=self.bias, kernel_size=self.kernel_size,
                               padding=self.padding)

            idx = g3_idx
            assert idx > 1
            self.g3 = MLPTemporal(
                channels=[num_joint*num_joint] + [out_channels2] * idx,
                kernel_sizes=[kernel_size2] * (idx-1) + [1],
                paddings=[kernel_size2//2] * (idx-1) + [0],
                dilations=[1] * idx,
                biases=[self.bias] * idx,
                residuals=[0] * idx,
                dropouts=[nn.Dropout2d] + [None] * (idx-1),
                activations=[nn.ReLU] * idx,
                normalizations=[nn.BatchNorm2d] * idx,
            )
            try:
                self.act1 = self.activation(dim=-1)
            except TypeError:
                self.act1 = self.activation()
            self.alpha = nn.Parameter(torch.zeros(1))

            sem_tem_emb_kwargs = dict(
                in_channels=num_segment,
                out_channels=num_joint*num_joint,
                bias=self.bias,
                dropout=nn.Dropout2d,
                activation=nn.ReLU,
                normalization=nn.BatchNorm2d,
                num_point=num_joint,
                mode=1
            )
            self.tem_onehot = OneHotTensor(num_segment, 1, mode=1)
            self.tem_embedding = Embedding(**sem_tem_emb_kwargs)

    def forward(self, x: Tensor, g: Optional[Tensor] = None) -> Tuple[Tensor]:
        if self.return_none:
            return None, None

        else:
            g1 = self.g1(x).permute(0, 3, 2, 1).contiguous()  # n,t,v,c
            g2 = self.g2(x).permute(0, 3, 1, 2).contiguous()  # n,t,c,v
            g12 = g1.matmul(g2)  # n,t,v,v
            g12 = self.act1(g12)  # n,t,v,v'

            tem_emb = self.tem_embedding(self.tem_onehot(x.shape[0]))
            x3 = rearrange(
                g12, 'n t i j -> n (i j) t').unsqueeze(2)  # n,vv,1,t
            g3 = self.g3(x3+tem_emb)  # n,c,1,t

            return g12, g3


class GCNSpatialGT6(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: int = 0,
                 activation: T1 = nn.Softmax,
                 g_proj_shared: bool = False,
                 num_joint: int = 25,
                 num_segment: int = 20,
                 **kwargs
                 ):
        super(GCNSpatialGT6, self).__init__(in_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            padding=padding,
                                            bias=bias,
                                            activation=activation)
        if self.kernel_size == 0:
            self.return_none = True

        else:
            self.return_none = False
            self.g1 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                           kernel_size=self.kernel_size, padding=self.padding)
            if g_proj_shared:
                self.g2 = self.g1
            else:
                self.g2 = Conv(self.in_channels, self.out_channels,
                               bias=self.bias, kernel_size=self.kernel_size,
                               padding=self.padding)
            try:
                self.act1 = self.activation(dim=-1)
            except TypeError:
                self.act1 = self.activation()
            self.alpha = nn.Parameter(torch.zeros(1))

            sem_tem_emb_kwargs = dict(
                in_channels=num_segment,
                out_channels=num_joint*num_joint,
                bias=self.bias,
                dropout=nn.Dropout2d,
                activation=nn.ReLU,
                normalization=nn.BatchNorm2d,
                num_point=num_joint,
                mode=1
            )
            self.tem_onehot = OneHotTensor(num_segment, 1, mode=1)
            self.tem_embedding = Embedding(**sem_tem_emb_kwargs)

    def forward(self, x: Tensor, g: Optional[Tensor] = None) -> Tuple[Tensor]:
        if self.return_none:
            return None, None

        else:
            g1 = self.g1(x).permute(0, 3, 2, 1).contiguous()  # n,t,v,c
            g2 = self.g2(x).permute(0, 3, 1, 2).contiguous()  # n,t,c,v
            g12 = g1.matmul(g2)  # n,t,v,v
            g12 = self.act1(g12)  # n,t,v,v'

            tem_emb = self.tem_embedding(self.tem_onehot(x.shape[0]))
            x3 = rearrange(
                g12, 'n t i j -> n (i j) t').unsqueeze(2)  # n,vv,1,t
            g3 = x3 + tem_emb

            return g12, g3


class GCNSpatialUnit(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: int = 0,
                 dropout: Optional[Type[PyTorchModule]] = None,
                 activation: T1 = nn.ReLU,
                 normalization: T1 = nn.BatchNorm2d,
                 prenorm: bool = False,
                 v_kernel_size: int = 0,
                 attn_mode: int = 0,
                 res_alpha: float = 1.0,
                 in_channels2: int = 128,
                 gt_mode: int = 0
                 ):
        super(GCNSpatialUnit, self).__init__(in_channels,
                                             out_channels,
                                             kernel_size=kernel_size,
                                             padding=padding,
                                             bias=bias,
                                             dropout=dropout,
                                             activation=activation,
                                             normalization=normalization,
                                             prenorm=prenorm)
        if res_alpha < 0:
            self.res_alpha = nn.Parameter(torch.ones(1))
        else:
            self.res_alpha = res_alpha

        self.gt_mode = gt_mode
        self.attn_mode = attn_mode

        if v_kernel_size > 0:
            self.w0 = Conv(self.in_channels,
                           self.in_channels,
                           kernel_size=v_kernel_size,
                           padding=v_kernel_size//2,
                           bias=self.bias)
        else:
            self.w0 = nn.Identity()

        # in original SGN bias for w1 is false
        if self.attn_mode != 3:
            self.w1 = Conv(self.in_channels, self.out_channels, bias=self.bias)

        if self.kernel_size > 0:
            if self.attn_mode == 10:
                self.w2 = Conv(in_channels2,
                               self.out_channels,
                               bias=self.bias,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
            else:
                self.w2 = Conv(self.in_channels,
                               self.out_channels,
                               bias=self.bias,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        else:
            self.w2 = null_fn

        # squeeze and excite
        if self.attn_mode == 2:
            self.w1 = Conv(self.out_channels//2, self.out_channels,
                           bias=self.bias)
            self.w3 = Conv(self.in_channels, self.out_channels//2,
                           bias=self.bias, activation=get_activation_fn('relu'))
            self.s = nn.Sigmoid()

        if self.prenorm:
            self.norm = nn.Identity()
        else:
            self.norm = self.normalization(self.out_channels)

        self.act = self.activation()
        self.drop = nn.Identity() if self.dropout is None else self.dropout()

    def forward(self,
                x: Tensor,
                g_list: Optional[Tuple[Tensor]] = None,
                y: Optional[Tensor] = None) -> Tensor:

        g = g_list[0]
        g1 = g_list[1]

        if self.gt_mode == 4:
            xg = g1*x
        else:
            xg = x

        x0 = self.w0(xg)
        # Original
        if self.attn_mode == 0:
            x1 = x0.permute(0, 3, 2, 1).contiguous()  # n,t,v,c
            x2 = g.matmul(x1)
            x3 = x2.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
            x4 = self.w1(x3)
            x5 = self.w2(xg) * self.res_alpha
            x6 = x4 + x5  # z + residual
        # Original, with y
        elif self.attn_mode == 10:
            x1 = x0.permute(0, 3, 2, 1).contiguous()  # n,t,v,c
            x2 = g.matmul(x1)
            x3 = x2.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
            x4 = self.w1(x3)
            x5 = self.w2(y) * self.res_alpha
            x6 = x4 + x5  # z + residual
        # 2 linear projections, no G
        elif self.attn_mode == 1:
            x1 = None
            x2 = None
            x3 = x0
            x4 = self.w1(x3)
            x5 = self.w2(xg) * self.res_alpha
            x6 = x4 + x5  # z + residual
        # SE instead of G
        elif self.attn_mode == 2:
            N, _, V, T = x0.shape
            # x1 = nn.AdaptiveAvgPool2d((1, T))  # n,c,1,t
            x1 = F.adaptive_avg_pool2d(x0, (1, T))
            x2 = self.w3(x1)
            x3 = self.w1(x2)
            x4 = self.s(x3).expand((N, -1, V, T))
            x5 = self.w2(xg) * self.res_alpha
            x6 = x4 + x5  # z + residual
        # 1 linear projection.
        elif self.attn_mode == 3:
            x1 = None
            x2 = None
            x3 = None
            x4 = None
            x5 = self.w2(xg)
            x6 = x5
        x7 = self.norm(x6)
        x8 = self.act(x7)
        x9 = self.drop(x8)
        return x9, {'x': x, 'x0': x0, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
                    'x5': x5, 'x6': x6, 'x7': x7, 'x8': x8, 'x9': x9}


class GCNSpatialBlock(PyTorchModule):
    def __init__(self,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: int = 0,
                 dropout: Optional[Type[PyTorchModule]] = None,
                 activation: T1 = nn.ReLU,
                 normalization: T1 = nn.BatchNorm2d,
                 gcn_dims: List[int] = [128, 256, 256],
                 gcn_residual: T3 = [0, 0, 0],
                 gcn_prenorm: bool = False,
                 gcn_v_kernel: int = 0,
                 gcn_ffn: Optional[float] = None,
                 gcn_attn_mode: int = 0,
                 g_proj_dim: T3 = 256,
                 g_kernel: int = 1,
                 g_proj_shared: bool = False,
                 g_activation: T1 = nn.Softmax,
                 g_weighted: int = 0,
                 g_num_segment: int = 20,
                 g_num_joint: int = 25,
                 g_res_alpha: float = 1.0,
                 gt_mode: int = 1,
                 gt_act: int = 1,
                 gt_g3_idx: int = 2,
                 gt_out_channels2: int = 512
                 ):
        super(GCNSpatialBlock, self).__init__()

        if gt_mode == 0:
            gcn_spa_gt_cls = GCNSpatialG
        elif gt_mode == 1:
            gcn_spa_gt_cls = GCNSpatialGT
        elif gt_mode == 2:
            gcn_spa_gt_cls = GCNSpatialGT2
        elif gt_mode == 3:
            gcn_spa_gt_cls = GCNSpatialGT3
        elif gt_mode == 4:
            gcn_spa_gt_cls = GCNSpatialGT4
        elif gt_mode == 5:
            gcn_spa_gt_cls = GCNSpatialGT5
        elif gt_mode == 6:
            gcn_spa_gt_cls = GCNSpatialGT6
        else:
            raise ValueError("Unknown gt_mode")

        self.num_blocks = len(gcn_dims) - 1
        self.g_shared = isinstance(g_proj_dim, int)
        if self.g_shared:
            self.gcn_g1 = gcn_spa_gt_cls(gcn_dims[0],
                                         g_proj_dim,
                                         bias=bias,
                                         kernel_size=g_kernel,
                                         padding=g_kernel//2,
                                         activation=g_activation,
                                         g_proj_shared=g_proj_shared,
                                         gt_activation=gt_act,
                                         num_segment=g_num_segment,
                                         num_joint=g_num_joint,
                                         g3_idx=gt_g3_idx,
                                         out_channels2=gt_out_channels2)
        else:
            for i in range(self.num_blocks):
                setattr(self, f'gcn_g{i+1}',
                        gcn_spa_gt_cls(gcn_dims[i],
                                       g_proj_dim[i],
                                       bias=bias,
                                       kernel_size=g_kernel,
                                       padding=g_kernel//2,
                                       activation=g_activation,
                                       g_proj_shared=g_proj_shared,
                                       gt_activation=gt_act,
                                       num_segment=g_num_segment,
                                       num_joint=g_num_joint,
                                       g3_idx=gt_g3_idx,
                                       out_channels2=gt_out_channels2))

        if g_weighted == 1:
            assert not self.g_shared
            self.g_weighted = g_weighted

        for i in range(self.num_blocks):
            setattr(self, f'gcn{i+1}',
                    GCNSpatialUnit(gcn_dims[i],
                                   gcn_dims[i+1],
                                   bias=bias,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   dropout=dropout,
                                   activation=activation,
                                   normalization=normalization,
                                   prenorm=gcn_prenorm,
                                   v_kernel_size=gcn_v_kernel,
                                   attn_mode=gcn_attn_mode,
                                   res_alpha=g_res_alpha,
                                   gt_mode=gt_mode))

        if gcn_prenorm:
            for i in range(self.num_blocks):
                setattr(self, f'gcn_prenorm{i+1}', normalization(gcn_dims[i]))

        if gcn_ffn is not None:
            for i in range(self.num_blocks):
                setattr(self, f'gcn_ffn{i+1}',
                        GCNSpatialFFN(gcn_dims[i+1],
                                      gcn_dims[i+1],
                                      bias=bias,
                                      activation=activation,
                                      normalization=normalization,
                                      multiplier=gcn_ffn))

        if isinstance(gcn_residual, list):
            assert len(gcn_residual) == self.num_blocks
            for i, r in enumerate(gcn_residual):
                setattr(self, f'gcn_res{i+1}',
                        res(r, gcn_dims[i], gcn_dims[i+1], bias))
            self.res = null_fn

        elif isinstance(gcn_residual, int):
            self.res = res(gcn_residual, gcn_dims[0], gcn_dims[-1], bias)

        else:
            raise ValueError("Unknown residual modes...")

    def forward(self, x: Tensor) -> Tuple[Tensor, list, list]:
        x0 = x
        g, gcn_list, fm_list = [], [], []
        for i in range(self.num_blocks):
            x1 = x

            if hasattr(self, f'gcn_prenorm{i+1}'):
                x1 = getattr(self, f'gcn_prenorm{i+1}')(x1)

            if len(g) == 0:
                g.append(getattr(self, f'gcn_g{i+1}')(x1))
            else:
                if not self.g_shared:
                    if hasattr(self, 'g_weighted'):
                        g.append(getattr(self, f'gcn_g{i+1}')(x1, g[-1]))
                    else:
                        g.append(getattr(self, f'gcn_g{i+1}')(x1))

            r = getattr(self, f'gcn_res{i+1}')(x)
            z, z_dict = getattr(self, f'gcn{i+1}')(x1, g[-1], x0)
            x = z + r

            if hasattr(self, f'gcn_ffn{i+1}'):
                x = getattr(self, f'gcn_ffn{i+1}')(x)

            fm_list.append(z_dict)
            gcn_list.append(x)

        x += self.res(x0)

        return x, g, gcn_list, fm_list


class GCNSpatialBlock2(PyTorchModule):
    def __init__(self,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: int = 0,
                 dropout: Optional[Type[PyTorchModule]] = None,
                 activation: T1 = nn.ReLU,
                 normalization: T1 = nn.BatchNorm2d,
                 gcn_dims_in: List[int] = [256, 128, 128],
                 gcn_dims: List[int] = [256, 256, 256],
                 gcn_residual: T3 = [0, 0, 0],
                 gcn_prenorm: bool = False,
                 gcn_v_kernel: int = 0,
                 gcn_ffn: Optional[float] = None,
                 gcn_attn_mode: int = 0,
                 g_proj_dim: T3 = 256,
                 g_kernel: int = 1,
                 g_proj_shared: bool = False,
                 g_activation: T1 = nn.Softmax,
                 g_weighted: int = 0,
                 g_num_segment: int = 20,
                 g_num_joint: int = 25,
                 g_res_alpha: float = 1.0,
                 gt_mode: int = 1,
                 gt_act: int = 1,
                 gt_g3_idx: int = 2
                 ):
        super(GCNSpatialBlock2, self).__init__()

        if gt_mode == 1:
            gcn_spa_gt_cls = GCNSpatialGT
        elif gt_mode == 2:
            gcn_spa_gt_cls = GCNSpatialGT2
        elif gt_mode == 3:
            gcn_spa_gt_cls = GCNSpatialGT3
        elif gt_mode == 4:
            gcn_spa_gt_cls = GCNSpatialGT4
        else:
            raise ValueError("Unknown gt_mode")

        self.num_blocks = len(gcn_dims) - 1
        self.g_shared = isinstance(g_proj_dim, int)
        if self.g_shared:
            self.gcn_g1 = gcn_spa_gt_cls(gcn_dims[0],
                                         g_proj_dim,
                                         bias=bias,
                                         kernel_size=g_kernel,
                                         padding=g_kernel//2,
                                         activation=g_activation,
                                         g_proj_shared=g_proj_shared,
                                         gt_activation=gt_act,
                                         num_segment=g_num_segment,
                                         num_joint=g_num_joint,
                                         g3_idx=gt_g3_idx)
        else:
            for i in range(self.num_blocks):
                setattr(self, f'gcn_g{i+1}',
                        gcn_spa_gt_cls(gcn_dims[i],
                                       g_proj_dim[i],
                                       bias=bias,
                                       kernel_size=g_kernel,
                                       padding=g_kernel//2,
                                       activation=g_activation,
                                       g_proj_shared=g_proj_shared,
                                       gt_activation=gt_act,
                                       num_segment=g_num_segment,
                                       num_joint=g_num_joint,
                                       g3_idx=gt_g3_idx))

        if g_weighted == 1:
            assert not self.g_shared
            self.g_weighted = g_weighted

        for i in range(self.num_blocks):
            setattr(self, f'gcn{i+1}',
                    GCNSpatialUnit(gcn_dims[i],
                                   gcn_dims[i+1],
                                   bias=bias,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   dropout=dropout,
                                   activation=activation,
                                   normalization=normalization,
                                   prenorm=gcn_prenorm,
                                   v_kernel_size=gcn_v_kernel,
                                   attn_mode=gcn_attn_mode,
                                   res_alpha=g_res_alpha,
                                   in_channels2=gcn_dims_in[i],
                                   gt_mode=gt_mode))

        if gcn_prenorm:
            for i in range(self.num_blocks):
                setattr(self, f'gcn_prenorm{i+1}', normalization(gcn_dims[i]))

        if gcn_ffn is not None:
            for i in range(self.num_blocks):
                setattr(self, f'gcn_ffn{i+1}',
                        GCNSpatialFFN(gcn_dims[i+1],
                                      gcn_dims[i+1],
                                      bias=bias,
                                      activation=activation,
                                      normalization=normalization,
                                      multiplier=gcn_ffn))

        if isinstance(gcn_residual, list):
            assert len(gcn_residual) == self.num_blocks
            for i, r in enumerate(gcn_residual):
                setattr(self, f'gcn_res{i+1}',
                        res(r, gcn_dims[i], gcn_dims[i+1], bias))
            self.res = null_fn

        elif isinstance(gcn_residual, int):
            self.res = res(gcn_residual, gcn_dims[0], gcn_dims[-1], bias)

        else:
            raise ValueError("Unknown residual modes...")

    def forward(self,
                x: Tensor,
                x_list: List[Tensor],
                g_attn: Optional[Tuple[Tensor]] = None,
                ) -> Tuple[Tensor, list, list]:

        # x_list: fm prior to GCN, high to low level
        assert len(x_list) == self.num_blocks

        x0 = x
        g, gcn_list, fm_list = [], [], []
        for i in range(self.num_blocks):
            x1 = x

            if hasattr(self, f'gcn_prenorm{i+1}'):
                x1 = getattr(self, f'gcn_prenorm{i+1}')(x1)

            if g_attn is None:
                if len(g) == 0:
                    g.append(getattr(self, f'gcn_g{i+1}')(x1))
                else:
                    if not self.g_shared:
                        if hasattr(self, 'g_weighted'):
                            g.append(getattr(self, f'gcn_g{i+1}')(x1, g[-1]))
                        else:
                            g.append(getattr(self, f'gcn_g{i+1}')(x1))

                r = getattr(self, f'gcn_res{i+1}')(x)
                z, z_dict = getattr(self, f'gcn{i+1}')(x1, g[-1], x_list[i])
                x = z + r

            else:
                r = getattr(self, f'gcn_res{i+1}')(x)
                z, z_dict = getattr(self, f'gcn{i+1}')(x1, g_attn, x_list[i])
                x = z + r

            if hasattr(self, f'gcn_ffn{i+1}'):
                x = getattr(self, f'gcn_ffn{i+1}')(x)

            fm_list.append(z_dict)
            gcn_list.append(x)

        x += self.res(x0)

        return x, g, gcn_list, fm_list
