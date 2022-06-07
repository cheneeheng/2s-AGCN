# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Taken from :
# https://github.com/microsoft/SGN/blob/master/model.py

# Code refractored

# Continue from on sgn_v7
# NO PARTS

# FREEZE 220427


import torch
from torch import nn
from torch import Tensor
from torch.nn import Module as PyTorchModule

try:
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
except ImportError:
    print("Warning: fvcore is not found")

import math
from typing import OrderedDict, Tuple, Optional, Union, Type, List, Any

from model.module import *
from model.module.layernorm import LayerNorm
from model.resource.common_ntu import *
from model.torch_utils import *

from utils.utils import *


T1 = Type[PyTorchModule]
T2 = List[Optional[Type[PyTorchModule]]]


class SGN(PyTorchModule):

    emb_modes = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    g_activation_fn = nn.Softmax
    # activation_fn = nn.ReLU
    # dropout_fn = nn.Dropout2d

    c1, c2, c3, c4 = c1, c2, c3, c4

    def __init__(self,
                 num_class: int = 60,
                 num_point: int = 25,
                 num_segment: int = 20,
                 in_channels: int = 3,
                 bias: int = 1,
                 dropout: float = 0.0,  # classifier
                 dropout2d: float = 0.0,  # the rest
                 c_multiplier: Union[int, float, list] = 1,
                 norm_type: str = 'bn-pre',
                 act_type: str = 'relu',

                 in_position: int = 1,
                 in_velocity: int = 1,

                 xpos_proj: int = 0,

                 sem_pos: int = 1,
                 sem_fra: int = 1,

                 sem_pos_fusion: int = 0,
                 sem_fra_fusion: int = 1,
                 dual_gcn_fusion: int = 0,

                 g_kernel: int = 1,
                 g_proj_shared: bool = False,
                 g_proj_dim: Union[List[int], int] = c3,  # c3
                 g_residual: Union[List[int], int] = [0, 0, 0],
                 gcn_t_kernel: int = 1,
                 gcn_dropout: float = 0.0,
                 gcn_dims: list = [c2, c3, c3],
                 gcn_ffn: int = 0,

                 gcn_tem: int = 0,
                 g_tem_kernel: int = 1,
                 g_tem_proj_shared: bool = False,
                 g_tem_proj_dim: Union[List[int], int] = c3,  # c3
                 g_tem_residual: Union[List[int], int] = [0, 0, 0],
                 gcn_tem_t_kernel: int = 1,
                 gcn_tem_dropout: float = 0.0,
                 gcn_tem_dims: list = [c2, c3, c3],
                 gcn_tem_ffn: int = 0,

                 t_g_kernel: int = 1,
                 t_g_proj_shared: bool = False,
                 t_g_proj_dim: Union[List[int], int] = c4,
                 t_g_residual: Union[List[int], int] = [0, 0, 0],
                 t_gcn_t_kernel: int = 1,
                 t_gcn_dropout: float = 0.0,
                 t_gcn_dims: list = [c3, c4, c4],
                 t_gcn_ffn: int = 0,

                 t_mode: int = 1,
                 t_kernel: int = 3,
                 t_maxpool_kwargs: Optional[dict] = None,
                 aspp: list = None,

                 spatial_maxpool: int = 1,
                 temporal_maxpool: int = 1,

                 ):
        super(SGN, self).__init__()

        self.num_class = num_class
        self.num_point = num_point
        self.num_segment = num_segment
        self.in_channels = in_channels
        self.bias = bias
        self.dropout_fn = lambda: nn.Dropout2d(dropout2d)

        if isinstance(c_multiplier, (int, float)):
            c_multiplier = [c_multiplier for _ in range(4)]
        elif isinstance(c_multiplier, list):
            c_multiplier = c_multiplier
        else:
            raise ValueError("Unknown c_multiplier type")
        self.c1 = to_int(self.c1 * c_multiplier[0])  # pos,vel,joint embed
        self.c2 = to_int(self.c2 * c_multiplier[1])  # G,gcn
        self.c3 = to_int(self.c3 * c_multiplier[2])  # gcn
        self.c4 = to_int(self.c4 * c_multiplier[3])  # gcn

        self.norm_type = norm_type
        if 'bn' in self.norm_type:
            self.normalization_fn = nn.BatchNorm2d
            self.normalization_fn_1d = nn.BatchNorm1d
        elif 'ln' in self.norm_type:
            self.normalization_fn = LayerNorm
            self.normalization_fn_1d = LayerNorm
        else:
            raise ValueError("Unknown norm_type ...")
        if 'pre' in self.norm_type:
            self.prenorm = True
        else:
            self.prenorm = False

        self.act_type = act_type
        assert self.act_type in ['relu', 'gelu']
        if self.act_type == 'relu':
            self.activation_fn = nn.ReLU
        elif self.act_type == 'gelu':
            self.activation_fn = nn.GELU

        self.in_position = in_position
        self.in_velocity = in_velocity
        assert self.in_position in self.emb_modes
        assert self.in_velocity in self.emb_modes

        self.xpos_proj = xpos_proj
        assert self.xpos_proj in self.emb_modes

        self.sem_pos = sem_pos
        self.sem_fra = sem_fra
        assert self.sem_pos in self.emb_modes
        assert self.sem_fra in self.emb_modes
        if self.in_position == 0 and self.sem_pos > 0:
            raise ValueError("in_position is 0 but sem_position is not")

        self.sem_pos_fusion = sem_pos_fusion
        self.sem_fra_fusion = sem_fra_fusion
        self.dual_gcn_fusion = dual_gcn_fusion
        # 0 = concat, 1 = add
        assert self.sem_pos_fusion in [0, 1]
        # 1 = add after GCN, 101 = add before GCN
        assert self.sem_fra_fusion in [1, 101]
        assert self.dual_gcn_fusion in [0, 1]

        self.g_kernel = g_kernel
        self.g_proj_shared = g_proj_shared
        self.g_proj_dim = g_proj_dim
        self.g_residual = g_residual
        self.gcn_t_kernel = gcn_t_kernel
        self.gcn_dropout_fn = lambda: nn.Dropout2d(gcn_dropout)
        self.gcn_dims = gcn_dims
        self.gcn_ffn = gcn_ffn

        self.gcn_tem = gcn_tem
        self.g_tem_kernel = g_tem_kernel
        self.g_tem_proj_shared = g_tem_proj_shared
        self.g_tem_proj_dim = g_tem_proj_dim
        self.g_tem_residual = g_tem_residual
        self.gcn_tem_t_kernel = gcn_tem_t_kernel
        self.gcn_tem_dropout_fn = lambda: nn.Dropout2d(gcn_tem_dropout)
        self.gcn_tem_dims = gcn_tem_dims
        self.gcn_tem_ffn = gcn_tem_ffn

        self.t_g_kernel = t_g_kernel
        self.t_g_proj_shared = t_g_proj_shared
        self.t_g_proj_dim = t_g_proj_dim
        self.t_g_residual = t_g_residual
        self.t_gcn_t_kernel = t_gcn_t_kernel
        self.t_gcn_dropout_fn = lambda: nn.Dropout2d(t_gcn_dropout)
        self.t_gcn_dims = t_gcn_dims
        self.t_gcn_ffn = t_gcn_ffn

        self.t_mode = t_mode
        self.t_kernel = t_kernel
        assert self.t_kernel >= 0
        self.t_maxpool_kwargs = t_maxpool_kwargs
        self.aspp = aspp

        self.spatial_maxpool = spatial_maxpool
        self.temporal_maxpool = temporal_maxpool
        assert self.spatial_maxpool in [0, 1, 2, 3]

        if self.sem_pos_fusion == 1:
            self.gcn_in_ch = self.c1
        elif self.sem_pos > 0:
            self.gcn_in_ch = self.c1 * 2
        else:
            self.gcn_in_ch = self.c1

        if self.gcn_tem == 1:
            if self.sem_pos > 0:
                self.gcn_tem_in_ch = self.c1 * 2
            else:
                self.gcn_tem_in_ch = self.c1
        elif self.gcn_tem == 2:
            if self.sem_pos > 0:
                self.gcn_tem_in_ch = self.c1 * self.num_point * 2
            else:
                self.gcn_tem_in_ch = self.c1 * self.num_point

        # Dynamic Representation -----------------------------------------------
        if self.in_position > 0:
            self.pos_embed = self.init_emb(mode=self.in_position,
                                           num_point=self.num_point,
                                           in_channels=self.in_channels,
                                           in_norm=self.normalization_fn_1d)
        if self.in_velocity > 0:
            self.vel_embed = self.init_emb(mode=self.in_velocity,
                                           num_point=self.num_point,
                                           in_channels=self.in_channels,
                                           in_norm=self.normalization_fn_1d)
        if self.in_position == 0 and self.in_velocity == 0:
            raise ValueError("Input args are faulty...")

        # Joint Embedding ------------------------------------------------------
        # Frame Embedding ------------------------------------------------------
        if self.sem_fra_fusion == 1:  # post gcn
            out_channels = self.c3
        elif self.sem_fra_fusion == 101:  # pre gcn
            out_channels = self.gcn_in_ch
        # Joint Embedding
        if self.sem_pos > 0:
            self.spa = OneHotTensor(self.num_point, self.num_segment, mode=0)
            self.spa_embed = self.init_emb(mode=self.sem_pos,
                                           num_point=self.num_point,
                                           in_channels=self.num_point)
        # Frame Embedding
        if self.sem_fra > 0:
            if self.gcn_tem > 0 and self.dual_gcn_fusion == 0:
                out_channels *= 2
            self.tem = OneHotTensor(self.num_segment, self.num_point, mode=1)
            self.tem_embed = self.init_emb(mode=self.sem_fra,
                                           num_point=self.num_point,
                                           in_channels=self.num_segment,
                                           out_channels=out_channels)

        # projection layer pre GCN
        if self.xpos_proj > 0:
            if self.sem_pos_fusion == 1:
                in_channels = self.c1
            else:
                in_channels = self.c1*2
            inter_channels = self.get_inter_channels(self.xpos_proj, self.c2)
            self.xpos_projection = self.init_emb(mode=self.xpos_proj,
                                                 num_point=self.num_point,
                                                 in_channels=in_channels,
                                                 out_channels=self.c2,
                                                 inter_channels=inter_channels)

        # Position Embedding ---------------------------------------------------
        # Frame embedding is a form of PE

        # GCN ------------------------------------------------------------------
        self.gcn_spatial = GCNSpatialBlock(
            0,
            0,
            kernel_size=self.gcn_t_kernel,
            padding=self.gcn_t_kernel//2,
            bias=self.bias,
            dropout=self.gcn_dropout_fn,
            activation=self.activation_fn,
            normalization=self.normalization_fn,
            gcn_dims=[self.gcn_in_ch] + self.gcn_dims,
            g_proj_dim=self.g_proj_dim,
            g_kernel=self.g_kernel,
            g_proj_shared=self.g_proj_shared,
            g_activation=self.g_activation_fn,
            g_residual=self.g_residual,
            ffn_mode=self.gcn_ffn
        )
        if self.gcn_tem > 0:
            self.gcn_temporal = GCNSpatialBlock(
                0,
                0,
                kernel_size=self.gcn_tem_t_kernel,
                padding=self.gcn_tem_t_kernel//2,
                bias=self.bias,
                dropout=self.gcn_tem_dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                gcn_dims=[self.gcn_tem_in_ch] + self.gcn_tem_dims,
                g_proj_dim=self.g_tem_proj_dim,
                g_kernel=self.g_tem_kernel,
                g_proj_shared=self.g_tem_proj_shared,
                g_activation=self.g_activation_fn,
                g_residual=self.g_tem_residual,
                ffn_mode=self.gcn_tem_ffn
            )

        # ASPP -----------------------------------------------------------------
        # Frame level module ---------------------------------------------------
        self.init_temporal_mlp()

        # 0=identity, 3= n,cv,1,t
        if self.spatial_maxpool in [0, 3]:
            self.smp = nn.Identity()
        elif self.spatial_maxpool == 1:
            self.smp = nn.AdaptiveMaxPool2d((1, self.num_segment))
        elif self.spatial_maxpool == 2:
            k = 0
            if self.in_position > 0 or self.in_velocity > 0:
                k += self.num_point
            if self.in_part > 0 or self.in_motion > 0:
                k += self.parts_len
            self.smp = nn.Conv2d(self.c3,
                                 self.c3,
                                 kernel_size=(k, 1),
                                 padding=(0, 0),
                                 bias=bool(self.bias))

        if self.temporal_maxpool in [0, 3]:
            self.tmp = nn.Identity()
        elif self.temporal_maxpool == 1:
            self.tmp = nn.AdaptiveMaxPool2d((1, 1))
        elif self.temporal_maxpool == 2:
            self.tmp = nn.Conv2d(self.c4,
                                 self.c4,
                                 kernel_size=(1, self.num_segment),
                                 padding=(0, 0),
                                 bias=bool(self.bias))

        self.do = nn.Dropout(dropout)

        if self.t_mode == 0:
            fc_in_ch = self.c3
        elif self.temporal_maxpool == 3:
            fc_in_ch = self.c4 * self.num_segment
        else:
            fc_in_ch = self.c4
        self.fc = nn.Linear(fc_in_ch, num_class)

        self.init_weight()

    def get_inter_channels(self, mode: int, ch: int) -> Union[list, int]:
        if mode == 3:
            return [ch, ch, ch]
        elif mode == 7:
            return []
        elif mode == 5:
            # inspired by the 4x dim in ffn of transformers
            return ch * 4
        elif mode == 8:
            return ch // 2
        else:
            return ch

    def pad_zeros(self, x: Tensor) -> Tensor:
        return torch.cat([x.new(*x.shape[:-1], 1).zero_(), x], dim=-1)

    def init_emb(self,
                 mode: int,
                 num_point: int,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 inter_channels: Optional[int] = None,
                 bias: Optional[int] = None,
                 dropout: Optional[T1] = None,
                 activation: Optional[T1] = None,
                 normalization: Optional[T1] = None,
                 in_norm: Optional[T1] = None) -> Module:
        inter_channels = inter_channels if inter_channels is not None else self.get_inter_channels(mode, self.c1)  # noqa
        out_channels = out_channels if out_channels is not None else self.c1
        bias = bias if bias is not None else self.bias
        dropout = dropout if dropout is not None else self.dropout_fn
        activation = activation if activation is not None else self.activation_fn  # noqa
        normalization = normalization if normalization is not None else self.normalization_fn  # noqa
        return Embedding(in_channels,
                         out_channels,
                         bias=bias,
                         dropout=dropout,
                         activation=activation,
                         normalization=normalization,
                         in_norm=in_norm,
                         inter_channels=inter_channels,
                         num_point=num_point,
                         mode=mode)

    def init_weight(self):
        """Follows the weight initialization from the original SGN codebase."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        init_zeros(self.gcn_spatial.gcn1.w1.block.conv.conv.weight)
        init_zeros(self.gcn_spatial.gcn2.w1.block.conv.conv.weight)
        init_zeros(self.gcn_spatial.gcn3.w1.block.conv.conv.weight)

    def init_temporal_mlp(self):
        _c3 = self.c3
        _c4 = self.c4
        if self.spatial_maxpool == 3:
            _c3 *= self.num_point
            assert self.t_mode in [9, 10]
        if self.gcn_tem > 0 and self.dual_gcn_fusion == 0:
            _c3 *= 2

        # aspp
        if self.aspp is None or len(self.aspp) == 0:
            self.aspp = nn.Identity()
        else:
            self.aspp = ASPP(_c3,
                             _c3,
                             bias=self.bias,
                             dilation=self.aspp,
                             dropout=self.dropout_fn,
                             activation=self.activation,
                             normalization=self.normalization_fn)
        # skip
        if self.t_mode == 0:
            self.cnn = nn.Identity()
        elif self.t_mode == 100:
            self.cnn = GCNSpatialBlock(
                0,
                0,
                kernel_size=self.t_gcn_t_kernel,
                padding=0,
                bias=self.bias,
                dropout=self.t_gcn_dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                gcn_dims=[_c3] + self.t_gcn_dims,
                g_proj_dim=self.t_g_proj_dim,
                g_kernel=self.t_g_kernel,
                g_proj_shared=self.t_g_proj_shared,
                g_activation=self.g_activation_fn,
                g_residual=self.t_g_residual,
                ffn_mode=self.t_gcn_ffn
            )
        elif self.t_mode in [101, 102]:
            if self.t_mode == 101:
                idx = 2
                channels = [_c3, _c3, _c4]
                kernel_sizes = [self.t_kernel, 1]
                paddings = [self.t_kernel//2, 0]
                residuals = [0 for _ in range(idx)]
                dropouts = [self.dropout_fn, None]
            elif self.t_mode == 102:
                idx = 2
                channels = [_c3, _c3, _c4]
                kernel_sizes = [self.t_kernel, 1]
                paddings = [self.t_kernel//2, 0]
                residuals = [1 for _ in range(idx)]
                dropouts = [self.dropout_fn, None]
            self.cnn = nn.Sequential(OrderedDict([
                ('GCN',
                    GCNSpatialBlock(
                        0,
                        0,
                        kernel_size=self.t_gcn_t_kernel,
                        padding=0,
                        bias=self.bias,
                        dropout=self.t_gcn_dropout_fn,
                        activation=self.activation_fn,
                        normalization=self.normalization_fn,
                        gcn_dims=[_c3] + self.t_gcn_dims,
                        g_proj_dim=self.t_g_proj_dim,
                        g_kernel=self.t_g_kernel,
                        g_proj_shared=self.t_g_proj_shared,
                        g_activation=self.g_activation_fn,
                        g_residual=self.t_g_residual,
                        ffn_mode=self.t_gcn_ffn,
                        return_g=False
                    )),
                ('MLP',
                    MLPTemporal(
                        channels=channels,
                        kernel_sizes=kernel_sizes,
                        paddings=paddings,
                        biases=[self.bias for _ in range(idx)],
                        residuals=residuals,
                        dropouts=dropouts,
                        activations=[self.activation_fn for _ in range(idx)],
                        normalizations=[
                            self.normalization_fn for _ in range(idx)],
                        maxpool_kwargs=self.t_maxpool_kwargs,
                        prenorm=self.prenorm
                    ))
            ]))
        else:
            # original sgn
            if self.t_mode == 1:
                idx = 2
                channels = [_c3, _c3, _c4]
                kernel_sizes = [self.t_kernel, 1]
                paddings = [self.t_kernel//2, 0]
                residuals = [0 for _ in range(idx)]
                dropouts = [self.dropout_fn, None]
            # original sgn with residual
            elif self.t_mode == 2:
                idx = 2
                channels = [_c3, _c3, _c4]
                kernel_sizes = [self.t_kernel, 1]
                paddings = [self.t_kernel//2, 0]
                residuals = [1 for _ in range(idx)]
                dropouts = [self.dropout_fn, None]
            # all 3x3
            elif self.t_mode == 3:
                idx = 2
                channels = [_c3, _c3, _c4]
                kernel_sizes = [self.t_kernel for _ in range(2)]
                paddings = [self.t_kernel//2 for _ in range(2)]
                residuals = [0 for _ in range(2)]
                dropouts = [self.dropout_fn, None]
            # all 3x3 with residual
            elif self.t_mode == 4:
                idx = 2
                channels = [_c3, _c3, _c4]
                kernel_sizes = [self.t_kernel for _ in range(2)]
                paddings = [self.t_kernel//2 for _ in range(2)]
                residuals = [1 for _ in range(2)]
                dropouts = [self.dropout_fn, None]
            # 3 layers
            elif self.t_mode == 5:
                idx = 3
                channels = [_c3, _c3, _c3, _c4]
                kernel_sizes = [self.t_kernel, 1, 1]
                paddings = [self.t_kernel//2, 0, 0]
                residuals = [0 for _ in range(3)]
                dropouts = [self.dropout_fn, None, None]
            # 3 layers with residual
            elif self.t_mode == 6:
                idx = 3
                channels = [_c3, _c3, _c3, _c4]
                kernel_sizes = [self.t_kernel, 1, 1]
                paddings = [self.t_kernel//2, 0, 0]
                residuals = [1 for _ in range(3)]
                dropouts = [self.dropout_fn, None, None]
            # original sgn + all dropout
            elif self.t_mode == 7:
                idx = 2
                channels = [_c3, _c3, _c4]
                kernel_sizes = [self.t_kernel, 1]
                paddings = [self.t_kernel//2, 0]
                residuals = [0 for _ in range(idx)]
                dropouts = [self.dropout_fn, self.dropout_fn]
            # original sgn + all dropout + residual
            elif self.t_mode == 8:
                idx = 2
                channels = [_c3, _c3, _c4]
                kernel_sizes = [self.t_kernel, 1]
                paddings = [self.t_kernel//2, 0]
                residuals = [1 for _ in range(idx)]
                dropouts = [self.dropout_fn, self.dropout_fn]
            # original sgn with quarter input channel
            elif self.t_mode == 9:
                idx = 2
                channels = [_c3, _c3//4, _c4]
                kernel_sizes = [self.t_kernel, 1]
                paddings = [self.t_kernel//2, 0]
                residuals = [0 for _ in range(idx)]
                dropouts = [self.dropout_fn, None]
            # original sgn with quarter input channel + residual
            elif self.t_mode == 10:
                idx = 2
                channels = [_c3, _c3//4, _c4]
                kernel_sizes = [self.t_kernel, 1]
                paddings = [self.t_kernel//2, 0]
                residuals = [1 for _ in range(idx)]
                dropouts = [self.dropout_fn, None]
            else:
                raise ValueError("Unknown t_mode...")
            self.cnn = MLPTemporal(
                channels=channels,
                kernel_sizes=kernel_sizes,
                paddings=paddings,
                biases=[self.bias for _ in range(idx)],
                residuals=residuals,
                dropouts=dropouts,
                activations=[self.activation_fn for _ in range(idx)],
                normalizations=[self.normalization_fn for _ in range(idx)],
                maxpool_kwargs=self.t_maxpool_kwargs,
                prenorm=self.prenorm
            )

    def get_dy1(self, x: Tensor):
        bs, step, dim = x.shape
        num_point = dim // 3
        x1 = x.view((bs, step, num_point, 3))  # n,t,v,c
        x = x1.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
        dif = x[:, :, :, 1:] - x[:, :, :, 0:-1]  # n,c,v,t-1
        dif = self.pad_zeros(dif)

        if self.in_position > 0 and self.in_velocity > 0:
            pos = self.pos_embed(x)
            dif = self.vel_embed(dif)
            dy1 = pos + dif  # n,c,v,t
        elif self.in_position > 0:
            dy1 = self.pos_embed(x)
        elif self.in_velocity > 0:
            dy1 = self.vel_embed(dif)
        else:
            dy1 = None

        return dy1

    def forward(self,
                x: Tensor,
                s: Optional[Tensor] = None
                ) -> Tuple[Tensor, list]:
        """Model forward pass.

        Args:
            x (Tensor): 3D joint in_position of a sequence of skeletons.
            s (Tensor): Subject id for each frame.

        Returns:
            y (Tensor): logits.
            g (Tensor): Attention matrix for GCN.
        """
        assert x.shape[-1] % 3 == 0, "Only support input of xyz only."

        # Dynamic Representation -----------------------------------------------
        dy1 = self.get_dy1(x)
        assert dy1 is not None

        # Joint and frame embeddings -------------------------------------------
        spa1, tem1 = None, None
        if self.sem_pos > 0:
            spa1 = self.spa_embed(self.spa(x.shape[0]))
        if self.sem_fra > 0:
            tem1 = self.tem_embed(self.tem(x.shape[0]))

        # Joint-level Module ---------------------------------------------------
        # xyz embeddings
        if spa1 is None:
            x_pos = dy1  # n,c,v,t
        elif self.sem_pos_fusion == 0:
            x_pos = torch.cat([dy1, spa1], 1)  # n,c,v,t
        elif self.sem_pos_fusion == 1:
            x_pos = dy1 + spa1  # n,c,v,t

        if hasattr(self, 'xpos_projection'):
            x_pos = self.xpos_projection(x_pos)

        # temporal fusion pre gcn
        if self.sem_fra > 0 and self.sem_fra_fusion == 101:
            x = x_pos + tem1
        else:
            x = x_pos

        # GCN ------------------------------------------------------------------
        s = x.shape

        x0, g0 = self.gcn_spatial(x)
        if self.gcn_tem == 1:
            x0_tem, g0_tem = self.gcn_temporal(x.transpose(-1, -2))
            x, g = [x0, x0_tem], [g0, g0_tem]
        elif self.gcn_tem == 2:
            x0_tem, g0_tem = self.gcn_temporal(x.reshape((s[0], -1, s[-1], 1)))
            x, g = [x0, x0_tem], [g0, g0_tem]
        else:
            x, g = [x0], [g0]

        # Frame-level Module ---------------------------------------------------
        # spatial fusion post gcn
        if self.gcn_tem == 1:
            x1 = x[0]
            x2 = x[1].transpose(-1, -2)
            if self.dual_gcn_fusion == 0:
                x = torch.cat([x1, x2], 1)
            elif self.dual_gcn_fusion == 1:
                x = x1 + x2
        elif self.gcn_tem == 2:
            x1 = x[0]
            x2 = x[1].reshape((s[0], -1, s[2], s[3]))
            if self.dual_gcn_fusion == 0:
                x = torch.cat([x1, x2], 1)
            elif self.dual_gcn_fusion == 1:
                x = x1 + x2
        else:
            x = x[0]

        # temporal fusion post gcn
        if self.sem_fra > 0 and self.sem_fra_fusion == 1:
            x = x + tem1

        x = self.smp(x)

        x = self.aspp(x)

        if self.t_mode == 100:
            x, _ = self.cnn(x.transpose(-1, -2))
            x = x.transpose(-1, -2)
        elif self.t_mode in [101, 102]:
            x = self.cnn.GCN(x.transpose(-1, -2))
            x = self.cnn.MLP(x.transpose(-1, -2))
        else:
            x = self.cnn(x)

        # Classification -------------------------------------------------------
        y = self.tmp(x)

        y = torch.flatten(y, 1)
        y = self.do(y)
        y = self.fc(y)

        return y, g


class OneHotTensor(PyTorchModule):
    def __init__(self, dim_eye: int, dim_length: int, mode: int):
        super(OneHotTensor, self).__init__()
        onehot = torch.eye(dim_eye, dim_eye)
        onehot = onehot.unsqueeze(0).unsqueeze(0)
        onehot = onehot.repeat(1, dim_length, 1, 1)
        if mode == 0:
            onehot = onehot.permute(0, 3, 2, 1)
        elif mode == 1:
            onehot = onehot.permute(0, 3, 1, 2)
        else:
            raise ValueError("Unknown mode")
        self.register_buffer("onehot", onehot)

    def forward(self, bs: int) -> Tensor:
        x = self.onehot.repeat(bs, 1, 1, 1)
        return x


class DataNorm(PyTorchModule):
    def __init__(self,
                 dim: int,
                 normalization: T1 = nn.BatchNorm1d):
        super(DataNorm, self).__init__()
        self.bn = normalization(dim)  # channel dim * num_point

    def forward(self, x: Tensor) -> Tensor:
        bs, _, num_point, step = x.shape  # n,c,v,t
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_point, step).contiguous()
        return x


class Embedding(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: int = 0,
                 dropout: T1 = nn.Dropout2d,
                 activation: T1 = nn.ReLU,
                 normalization: T1 = nn.BatchNorm2d,
                 in_norm: Optional[T1] = None,
                 inter_channels: Union[list, int] = 0,
                 num_point: int = 25,
                 mode: int = 1
                 ):
        super(Embedding, self).__init__(in_channels,
                                        out_channels,
                                        bias=bias,
                                        dropout=dropout,
                                        activation=activation,
                                        normalization=normalization)
        assert mode in [1, 2, 3, 4, 5, 6, 7, 8]
        self.mode = mode

        if in_norm is not None:
            self.norm = DataNorm(self.in_channels * num_point, in_norm)
        else:
            self.norm = nn.Identity()

        if self.mode in [1, 4, 5, 6, 8]:
            # 1=original, 4=dropout, 5=higher inter, 6=residual, 8=lower inter
            if self.mode != 4:
                self.dropout = None
            self.cnn1 = Conv(self.in_channels,
                             inter_channels,
                             bias=self.bias,
                             activation=self.activation,
                             dropout=self.dropout)
            self.cnn2 = Conv(inter_channels,
                             self.out_channels,
                             bias=self.bias,
                             activation=self.activation)
            if self.mode == 6:
                if self.in_channels == inter_channels:
                    self.res1 = nn.Identity()
                else:
                    self.res1 = Conv(self.in_channels, inter_channels,
                                     bias=self.bias)
                if inter_channels == self.out_channels:
                    self.res2 = nn.Identity()
                else:
                    self.res2 = Conv(inter_channels, self.out_channels,
                                     bias=self.bias)
            else:
                self.res1 = null_fn
                self.res2 = null_fn

        elif self.mode == 2:
            # bert style (only if we have ont hot vector)
            self.cnn1 = Conv(self.in_channels,
                             self.out_channels,
                             bias=self.bias,
                             normalization=lambda: self.normalization(
                                 self.out_channels),
                             dropout=self.dropout)

        elif self.mode == 3 or self.mode == 7:
            assert isinstance(inter_channels, list)
            layer_channels = \
                [self.in_channels] + inter_channels + [self.out_channels]
            self.num_layers = len(layer_channels)-1
            for i in range(self.num_layers):
                setattr(self, f'cnn{i+1}', Conv(layer_channels[i],
                                                layer_channels[i+1],
                                                bias=self.bias,
                                                activation=self.activation))

    def forward(self, x: Tensor) -> Tensor:
        # x: n,c,v,t
        x = self.norm(x)
        if self.mode in [1, 4, 5, 6, 8]:
            x = self.cnn1(x) + self.res1(x)
            x = self.cnn2(x) + self.res2(x)
        elif self.mode == 2:
            x = self.cnn1(x)
        elif self.mode in [3, 7]:
            for i in range(self.num_layers):
                x = getattr(self, f'cnn{i+1}')(x)
        return x


class MLPTemporal(PyTorchModule):
    def __init__(self,
                 channels: List[int],
                 kernel_sizes: List[int] = [3, 1],
                 paddings: List[int] = [1, 0],
                 biases: List[int] = [0, 0],
                 residuals: List[int] = [0, 0],
                 dropouts: T2 = [nn.Dropout2d, None],
                 activations: T2 = [nn.ReLU, nn.ReLU],
                 normalizations: T2 = [nn.BatchNorm2d, nn.BatchNorm2d],
                 maxpool_kwargs: Optional[dict] = None,
                 residual: int = 0,
                 prenorm: bool = False
                 ):
        super(MLPTemporal, self).__init__()
        if maxpool_kwargs is not None:
            self.pool = nn.MaxPool2d(**maxpool_kwargs)
        else:
            self.pool = nn.Identity()
        if residual == 0:
            self.res = null_fn
        elif residual == 1:
            if channels[0] == channels[-1]:
                self.res = nn.Identity()
            else:
                self.res = Conv(channels[0], channels[-1], bias=biases[0])
        else:
            raise ValueError('Unknown residual mode...')
        self.num_layers = len(channels) - 1
        for i in range(self.num_layers):
            if prenorm:
                def norm_func(): return normalizations[i](channels[i])
            else:
                def norm_func(): return normalizations[i](channels[i+1])
            setattr(self,
                    f'cnn{i+1}',
                    Conv(channels[i],
                         channels[i+1],
                         kernel_size=kernel_sizes[i],
                         padding=paddings[i],
                         bias=biases[i],
                         activation=activations[i],
                         normalization=norm_func,
                         dropout=dropouts[i],
                         prenorm=prenorm)
                    )
            if residuals[i] == 0:
                setattr(self, f'res{i+1}', null_fn)
            elif residuals[i] == 1:
                if channels[i] == channels[i+1]:
                    setattr(self, f'res{i+1}', nn.Identity())
                else:
                    setattr(self, f'res{i+1}', Conv(channels[i], channels[i+1],
                                                    bias=biases[i]))
            else:
                raise ValueError('Unknown residual mode...')

    def forward(self, x: Tensor) -> Tensor:
        # x: n,c,v,t ; v=1 due to SMP
        x0 = x
        x = self.pool(x)
        for i in range(self.num_layers):
            x = getattr(self, f'cnn{i+1}')(x) + getattr(self, f'res{i+1}')(x)
        x += self.res(x0)
        return x


class GCNSpatialG(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: int = 0,
                 activation: T1 = nn.Softmax,
                 normalization: Optional[T1] = None,
                 prenorm: bool = False,
                 g_proj_shared: bool = False,
                 ):
        super(GCNSpatialG, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=bias,
                                          activation=activation,
                                          normalization=normalization,
                                          prenorm=prenorm)
        if self.prenorm:
            self.norm = self.normalization(self.in_channels)
        self.g1 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                       kernel_size=self.kernel_size, padding=self.padding)
        if g_proj_shared:
            self.g2 = self.g1
        else:
            self.g2 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                           kernel_size=self.kernel_size, padding=self.padding)
        self.act = self.activation(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        if self.prenorm:
            x = self.norm(x)
        g1 = self.g1(x).permute(0, 3, 2, 1).contiguous()  # n,t,v,c
        g2 = self.g2(x).permute(0, 3, 1, 2).contiguous()  # n,t,c,v
        g3 = g1.matmul(g2)  # n,t,v,v
        g4 = self.act(g3)
        return g4


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
                 prenorm: bool = False
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
        if self.prenorm:
            self.norm = self.normalization(self.in_channels)
        else:
            self.norm = self.normalization(self.out_channels)
        self.act = self.activation()
        self.drop = nn.Identity() if self.dropout is None else self.dropout()
        self.w1 = Conv(self.in_channels, self.out_channels, bias=self.bias)
        self.w2 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                       kernel_size=self.kernel_size, padding=self.padding)

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        if self.prenorm:
            x = self.norm(x)
        x1 = x.permute(0, 3, 2, 1).contiguous()  # n,t,v,c
        x1 = g.matmul(x1)
        x1 = x1.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
        x1 = self.w1(x1) + self.w2(x)  # z + residual
        if not self.prenorm:
            x1 = self.norm(x1)
        x1 = self.act(x1)
        x1 = self.drop(x1)
        return x1


class GCNSpatialBlock(Module):
    def __init__(self,
                 *args,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: int = 0,
                 dropout: Optional[Type[PyTorchModule]] = None,
                 activation: T1 = nn.ReLU,
                 normalization: T1 = nn.BatchNorm2d,
                 prenorm: bool = False,
                 gcn_dims: List[int],
                 g_proj_dim: Union[List[int], int],
                 g_kernel: int = 1,
                 g_proj_shared: bool = False,
                 g_activation: T1 = nn.Softmax,
                 g_residual: Union[List[int], int] = [0, 0, 0],
                 ffn_mode: int = 0,
                 return_g: bool = True
                 ):
        super(GCNSpatialBlock, self).__init__(*args,
                                              kernel_size=kernel_size,
                                              padding=padding,
                                              bias=bias,
                                              dropout=dropout,
                                              activation=activation,
                                              normalization=normalization,
                                              prenorm=prenorm)
        self.return_g = return_g
        self.num_blocks = len(gcn_dims) - 1
        self.g_shared = isinstance(g_proj_dim, int)
        if self.g_shared:
            self.gcn_g = GCNSpatialG(gcn_dims[0],
                                     g_proj_dim,
                                     bias=self.bias,
                                     kernel_size=g_kernel,
                                     padding=g_kernel//2,
                                     activation=g_activation,
                                     normalization=self.normalization,
                                     prenorm=self.prenorm,
                                     g_proj_shared=g_proj_shared)
        else:
            for i in range(self.num_blocks):
                setattr(self, f'gcn_g{i+1}',
                        GCNSpatialG(gcn_dims[i],
                                    g_proj_dim[i],
                                    bias=self.bias,
                                    kernel_size=g_kernel,
                                    padding=g_kernel//2,
                                    activation=g_activation,
                                    normalization=self.normalization,
                                    prenorm=self.prenorm,
                                    g_proj_shared=g_proj_shared))

        for i in range(self.num_blocks):
            setattr(self, f'gcn{i+1}',
                    GCNSpatialUnit(gcn_dims[i],
                                   gcn_dims[i+1],
                                   bias=self.bias,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   dropout=self.dropout,
                                   activation=self.activation,
                                   normalization=self.normalization,
                                   prenorm=self.prenorm))

        self.res = null_fn
        for i in range(self.num_blocks):
            setattr(self, f'res{i+1}', null_fn)

        if isinstance(g_residual, list):
            assert len(g_residual) == self.num_blocks
            for i, r in enumerate(g_residual):
                if r == 0:
                    setattr(self, f'res{i+1}', null_fn)
                elif r == 1:
                    if gcn_dims[i] == gcn_dims[i+1]:
                        setattr(self, f'res{i+1}', nn.Identity())
                    else:
                        setattr(self, f'res{i+1}', Conv(gcn_dims[i],
                                                        gcn_dims[i+1],
                                                        bias=self.bias))
                else:
                    raise ValueError("Unknown residual modes...")

        elif isinstance(g_residual, int):
            if g_residual == 1:
                if gcn_dims[0] == gcn_dims[-1]:
                    self.res = nn.Identity()
                else:
                    self.res = Conv(gcn_dims[0], gcn_dims[-1], bias=self.bias)
            else:
                raise ValueError("Unknown residual modes...")

        else:
            raise ValueError("Unknown residual modes...")

        if ffn_mode > 0:
            for i in range(self.num_blocks):
                if ffn_mode == 1:
                    idx = 2
                    channels = [gcn_dims[i+1], gcn_dims[i+1], gcn_dims[i+1]]
                    kernel_sizes = [1, 1]
                    paddings = [0, 0]
                    residuals = [0, 0]
                    dropouts = [self.dropout, None]
                    residual = 0
                elif ffn_mode == 2:
                    idx = 2
                    channels = [gcn_dims[i+1], gcn_dims[i+1], gcn_dims[i+1]]
                    kernel_sizes = [3, 1]
                    paddings = [1, 0]
                    residuals = [0, 0]
                    dropouts = [self.dropout, None]
                    residual = 0
                elif ffn_mode == 3:
                    idx = 2
                    channels = [gcn_dims[i+1], gcn_dims[i+1], gcn_dims[i+1]]
                    kernel_sizes = [1, 1]
                    paddings = [0, 0]
                    residuals = [0, 0]
                    dropouts = [self.dropout, None]
                    residual = 1
                elif ffn_mode == 4:
                    idx = 2
                    channels = [gcn_dims[i+1], gcn_dims[i+1], gcn_dims[i+1]]
                    kernel_sizes = [3, 1]
                    paddings = [1, 0]
                    residuals = [0, 0]
                    dropouts = [self.dropout, None]
                    residual = 1
                elif ffn_mode == 5:
                    idx = 2
                    channels = [gcn_dims[i+1], gcn_dims[i+1]*4, gcn_dims[i+1]]
                    kernel_sizes = [1, 1]
                    paddings = [0, 0]
                    residuals = [0, 0]
                    dropouts = [self.dropout, None]
                    residual = 1
                elif ffn_mode == 6:
                    idx = 2
                    channels = [gcn_dims[i+1], gcn_dims[i+1]*4, gcn_dims[i+1]]
                    kernel_sizes = [3, 1]
                    paddings = [1, 0]
                    residuals = [0, 0]
                    dropouts = [self.dropout, None]
                    residual = 1
                setattr(self,
                        f'ffn{i+1}',
                        MLPTemporal(
                            channels=channels,
                            kernel_sizes=kernel_sizes,
                            paddings=paddings,
                            biases=[self.bias for _ in range(idx)],
                            residuals=residuals,
                            dropouts=dropouts,
                            activations=[
                                self.activation for _ in range(idx)],
                            normalizations=[
                                self.normalization for _ in range(idx)],
                            maxpool_kwargs=None,
                            residual=residual,
                            prenorm=self.prenorm)
                        )
        else:
            for i in range(self.num_blocks):
                setattr(self, f'ffn{i+1}', nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        x0 = x
        if self.g_shared:
            g = self.gcn_g(x)
            for i in range(self.num_blocks):
                x = getattr(self, f'gcn{i+1}')(x, g) + \
                    getattr(self, f'res{i+1}')(x)
                x = getattr(self, f'ffn{i+1}')(x)
        else:
            g = []
            for i in range(self.num_blocks):
                g1 = getattr(self, f'gcn_g{i+1}')(x)
                g.append(g1)
                x = getattr(self, f'gcn{i+1}')(x, g1) + \
                    getattr(self, f'res{i+1}')(x)
                x = getattr(self, f'ffn{i+1}')(x)
        x += self.res(x0)
        if self.return_g:
            return x, g
        else:
            return x


if __name__ == '__main__':

    batch_size = 64

    inputs = torch.ones(batch_size, 20, 75)
    subjects = torch.ones(batch_size, 20, 1)

    model = SGN(num_segment=20,
                dual_gcn_fusion=1,
                gcn_tem=1,
                gcn_tem_dims=[128, 256, 256],
                )
    model(inputs, subjects)
    # print(model)

    # try:
    #     flops = FlopCountAnalysis(model, inputs)
    #     # print(flops.total())
    #     # print(flops.by_module_and_operator())
    #     print(flop_count_table(flops))
    # except NameError:
    #     print("Warning: fvcore is not found")
