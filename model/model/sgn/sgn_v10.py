# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Taken from :
# https://github.com/microsoft/SGN/blob/master/model.py

# Code refractored

# Continue from on sgn_v9
# NO PARTS


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

from utils.utils import *


T1 = Type[PyTorchModule]
T2 = List[Optional[Type[PyTorchModule]]]
T3 = Union[List[int], int]


class Identity(PyTorchModule):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, *args):
        return args[0]


def null_fn(x: Any) -> int:
    return 0


def init_0(x: Tensor):
    return nn.init.constant_(x, 0)


def pad_zeros(x: Tensor) -> Tensor:
    return torch.cat([x.new(*x.shape[:-1], 1).zero_(), x], dim=-1)


def get_inter_channels(mode: int, ch: int) -> Union[list, int]:
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


def fuse(x1: Tensor, x2: Tensor, mode: int) -> Tensor:
    if mode == 0:
        return torch.cat([x1, x2], 1)
    elif mode == 1:
        return x1 + x2
    else:
        raise ValueError('Unknown input emb fusion arg')


def activation_fn(act_type: str) -> Type[PyTorchModule]:
    if act_type == 'relu':
        return nn.ReLU
    elif act_type == 'gelu':
        return nn.GELU
    else:
        raise ValueError("Unknown act_type ...")


def normalization_fn(norm_type: str) -> Tuple[Type[PyTorchModule],
                                              Type[PyTorchModule]]:
    if 'bn' in norm_type:
        return nn.BatchNorm2d, nn.BatchNorm1d
    elif 'ln' in norm_type:
        return LayerNorm, LayerNorm
    else:
        raise ValueError("Unknown norm_type ...")


class SGN(PyTorchModule):

    # CONSTANTS
    ffn_mode = [0, 1, 2, 3, 101, 102, 103, 104, 201]
    emb_modes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    c1, c2, c3, c4 = c1, c2, c3, c4
    g_activation_fn = nn.Softmax

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

                 in_pos: int = 1,
                 in_vel: int = 1,

                 sem_pos: int = 1,
                 sem_fra: int = 1,
                 sem_pos_fusion: int = 0,
                 sem_fra_fusion: int = 1,
                 sem_fra_location: int = 0,

                 x_emb_proj: int = 0,

                 gcn_list: List[str] = ['spa'],
                 gcn_tem: int = 0,
                 gcn_fusion: int = 0,

                 gcn_spa_g_kernel: int = 1,
                 gcn_spa_g_proj_shared: bool = False,
                 gcn_spa_g_proj_dim: Optional[T3] = None,  # c3
                 gcn_spa_gcn_residual: T3 = [0, 0, 0],
                 gcn_spa_prenorm: bool = True,
                 gcn_spa_t_kernel: int = 1,
                 gcn_spa_dropout: float = 0.0,
                 gcn_spa_dims: Optional[list] = None,  # [c2, c3, c3],
                 gcn_spa_ffn: int = 1,
                 gcn_spa_ffn_prenorm: bool = False,

                 gcn_tem_g_kernel: int = 1,
                 gcn_tem_g_proj_shared: bool = False,
                 gcn_tem_g_proj_dim: Optional[T3] = None,  # c3
                 gcn_tem_gcn_residual: T3 = [0, 0, 0],
                 gcn_tem_prenorm: bool = True,
                 gcn_tem_t_kernel: int = 1,
                 gcn_tem_dropout: float = 0.0,
                 gcn_tem_dims: Optional[list] = None,  # [c2, c3, c3],
                 gcn_tem_ffn: int = 1,
                 gcn_tem_ffn_prenorm: bool = False,

                 t_g_kernel: int = 1,
                 t_g_proj_shared: bool = False,
                 t_g_proj_dim: Optional[T3] = None,  # c4,
                 t_gcn_residual: T3 = [0, 0, 0],
                 t_gcn_t_kernel: int = 1,
                 t_gcn_dropout: float = 0.0,
                 t_gcn_dims: Optional[list] = None,  # [c3, c4, c4],
                 t_gcn_ffn: int = 0,
                 t_gcn_prenorm: bool = False,

                 spatial_maxpool: int = 1,
                 temporal_maxpool: int = 1,

                 aspp_rates: list = None,
                 t_mode: int = 1,
                 t_kernel: int = 3,
                 t_maxpool_kwargs: Optional[dict] = None,

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

        (self.normalization_fn,
         self.normalization_fn_1d) = normalization_fn(norm_type)
        self.prenorm = True if 'pre' in norm_type else False

        self.activation_fn = activation_fn(act_type)

        self.in_pos = in_pos
        self.in_vel = in_vel
        self.sem_pos = sem_pos
        self.sem_fra = sem_fra
        self.x_emb_proj = x_emb_proj
        assert self.in_pos in self.emb_modes
        assert self.in_vel in self.emb_modes
        assert self.sem_pos in self.emb_modes
        assert self.sem_fra in self.emb_modes
        assert self.x_emb_proj in self.emb_modes
        if self.in_pos == 0 and self.sem_pos > 0:
            raise ValueError("in_pos is 0 but sem_position is not")

        # 0: concat, 1: sum
        self.sem_pos_fusion = sem_pos_fusion
        # 0: concat, 1: sum
        self.sem_fra_fusion = sem_fra_fusion
        # 0 = add after GCN, 1 = add before GCN
        self.sem_fra_location = sem_fra_location
        assert self.sem_fra_location in [0, 1]

        # spa, tem
        self.gcn_list = gcn_list
        self.gcn_fusion = gcn_fusion
        self.gcn_tem = gcn_tem
        # 0 = swap axis, 1 = merge vc
        assert self.gcn_tem in [0, 1]

        if self.sem_pos == 0:
            self.gcn_in_ch = self.c1
        else:
            if self.sem_pos_fusion == 0:
                self.gcn_in_ch = self.c1 * 2
            elif self.sem_pos_fusion == 1:
                self.gcn_in_ch = self.c1

        # Spatial GCN
        if gcn_spa_dims is None:
            gcn_spa_dims = [self.c2, self.c3, self.c3]
        if gcn_spa_g_proj_dim is None:
            gcn_spa_g_proj_dim = self.c3
        gcn_spatial_kwargs = dict(
            kernel_size=gcn_spa_t_kernel,
            padding=gcn_spa_t_kernel//2,
            dropout=lambda: nn.Dropout2d(gcn_spa_dropout),
            gcn_dims=[self.gcn_in_ch]+gcn_spa_dims,
            gcn_residual=gcn_spa_gcn_residual,
            gcn_prenorm=gcn_spa_prenorm,
            g_proj_dim=gcn_spa_g_proj_dim,
            g_kernel=gcn_spa_g_kernel,
            g_proj_shared=gcn_spa_g_proj_shared,
            ffn_mode=gcn_spa_ffn,
            ffn_prenorm=gcn_spa_ffn_prenorm,
        )
        assert gcn_spa_ffn in self.ffn_mode

        # Temporal GCN
        if gcn_tem_dims is None:
            gcn_tem_dims = [self.c2, self.c3, self.c3]
        gcn_dims = [self.gcn_in_ch] + gcn_tem_dims
        if self.gcn_tem == 1:
            gcn_dims = [i*self.num_point for i in gcn_dims]
        if gcn_tem_g_proj_dim is None:
            gcn_tem_g_proj_dim = self.c3
        gcn_temporal_kwargs = dict(
            kernel_size=gcn_tem_t_kernel,
            padding=gcn_tem_t_kernel//2,
            dropout=lambda: nn.Dropout2d(gcn_tem_dropout),
            gcn_dims=gcn_dims,
            gcn_residual=gcn_tem_gcn_residual,
            gcn_prenorm=gcn_tem_prenorm,
            g_proj_dim=gcn_tem_g_proj_dim,  # noqa
            g_kernel=gcn_tem_g_kernel,
            g_proj_shared=gcn_tem_g_proj_shared,
            ffn_mode=gcn_tem_ffn,
            ffn_prenorm=gcn_tem_ffn_prenorm,
        )
        assert gcn_tem_ffn in self.ffn_mode

        # GCN in temporal mlp
        if t_g_proj_dim is None:
            t_g_proj_dim = self.c4
        if t_gcn_dims is None:
            t_gcn_dims = [self.c3, self.c4, self.c4]
        self.t_gcn_kwargs = dict(
            kernel_size=t_gcn_t_kernel,
            padding=t_gcn_t_kernel//2,
            dropout=lambda: nn.Dropout2d(t_gcn_dropout),
            gcn_dims=t_gcn_dims,
            gcn_residual=t_gcn_residual,
            gcn_prenorm=t_gcn_prenorm,
            g_proj_dim=t_g_proj_dim,
            g_kernel=t_g_kernel,
            g_proj_shared=t_g_proj_shared,
            ffn_mode=t_gcn_ffn,
        )
        assert t_gcn_ffn in self.ffn_mode

        # 0 no pool, 1 pool, 2 projection, 3 no pool but merge channels
        self.spatial_maxpool = spatial_maxpool
        self.temporal_maxpool = temporal_maxpool
        assert self.spatial_maxpool in [0, 1, 2, 3]

        self.t_mode = t_mode
        self.t_kernel = t_kernel
        assert self.t_kernel >= 0
        self.t_maxpool_kwargs = t_maxpool_kwargs
        self.aspp_rates = aspp_rates  # dilation rates

        # Dynamic Representation -----------------------------------------------
        self.feature_extractor = FeatureExtractor(
            in_pos=self.in_pos,
            in_vel=self.in_vel,
            in_pos_kwargs=dict(
                in_channels=self.in_channels,
                out_channels=self.c1,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                in_norm=self.normalization_fn_1d,
                inter_channels=get_inter_channels(self.in_pos, self.c1),
                num_point=self.num_point,
                mode=self.in_pos
            ),
            in_vel_kwargs=dict(
                in_channels=self.in_channels,
                out_channels=self.c1,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                in_norm=self.normalization_fn_1d,
                inter_channels=get_inter_channels(self.in_pos, self.c1),
                num_point=self.num_point,
                mode=self.in_pos
            )
        )

        # Joint and frame embeddings -------------------------------------------
        # post gcn
        if self.sem_fra_location == 0:
            out_channels = self.c3
        # pre gcn
        elif self.sem_fra_location == 1:
            out_channels = self.gcn_in_ch

        # dual gcn + concat
        if self.sem_fra > 0:
            if len(self.gcn_list) == 2 and self.gcn_fusion == 0:
                out_channels *= 2

        self.semantic_embedding = SemanticEmbedding(
            num_point=self.num_point,
            num_segment=self.num_segment,
            sem_pos=self.sem_pos,
            sem_fra=self.sem_fra,
            spa_embed_kwargs=dict(
                in_channels=self.num_point,
                out_channels=self.c1,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                in_norm=self.normalization_fn_1d,
                inter_channels=get_inter_channels(self.sem_pos, self.c1),
                num_point=self.num_point,
                mode=self.sem_pos
            ),
            tem_embed_kwargs=dict(
                in_channels=self.num_segment,
                out_channels=out_channels,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                in_norm=self.normalization_fn_1d,
                inter_channels=get_inter_channels(self.sem_fra, self.c1),
                num_point=self.num_point,
                mode=self.sem_fra
            )
        )

        # projection layer pre GCN
        if self.x_emb_proj > 0:
            self.x_emb_projection = Embedding(
                in_channels=self.gcn_in_ch,
                out_channels=self.gcn_in_ch,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                in_norm=self.normalization_fn_1d,
                inter_channels=get_inter_channels(self.x_emb_proj, self.c2),
                num_point=self.num_point,
                mode=self.x_emb_proj
            )

        # GCN ------------------------------------------------------------------
        if 'spa' in self.gcn_list:
            self.gcn_spatial = GCNSpatialBlock(
                0,
                0,
                bias=self.bias,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                g_activation=self.g_activation_fn,
                **gcn_spatial_kwargs
            )
        if 'tem' in self.gcn_list:
            self.gcn_temporal = GCNSpatialBlock(
                0,
                0,
                bias=self.bias,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                g_activation=self.g_activation_fn,
                **gcn_temporal_kwargs
            )
        if 'dual' in self.gcn_list:
            self.gcn_dual = DualGCNSpatialBlock(self.gcn_spatial,
                                                self.gcn_temporal)

        # Frame level module ---------------------------------------------------
        # aspp + mlp
        _c3 = self.c3
        _c4 = self.c4
        # no pooling, just cv merge
        if self.spatial_maxpool == 3:
            _c3 *= self.num_point
            assert self.t_mode in [9, 10]
        # dual gcn
        if len(self.gcn_list) == 2 and self.gcn_fusion == 0:
            _c3 *= 2

        self.tem_mlp = MLPTemporalBranch(
            in_channels=_c3,
            out_channels=_c4,
            bias=self.bias,
            dropout=self.dropout_fn,
            activation=self.activation_fn,
            normalization=self.normalization_fn,
            prenorm=self.prenorm,
            g_activation_fn=self.g_activation_fn,
            aspp_rates=self.aspp_rates,
            t_mode=self.t_mode,
            t_kernel=self.t_kernel,
            t_maxpool_kwargs=self.t_maxpool_kwargs,
            t_gcn_kwargs=self.t_gcn_kwargs
        )

        # 0=nn.Identity, 3= n,cv,1,t
        if self.spatial_maxpool == 0:
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
        elif self.spatial_maxpool == 3:
            # n,cv,1,t
            self.smp = lambda z: z.reshape((z.shape[0], -1, 1, z.shape[-1]))
        else:
            raise ValueError("Unknown spatial_maxpool")

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

        # Classifier -----------------------------------------------------------
        self.fc_dropout = nn.Dropout(dropout)
        if self.t_mode == 0:
            fc_in_ch = self.c3
        elif self.spatial_maxpool == 0 and self.temporal_maxpool == 0:
            fc_in_ch = self.c4 * self.num_segment * self.num_point
        elif self.temporal_maxpool in [0, 3]:
            fc_in_ch = self.c4 * self.num_segment
        else:
            fc_in_ch = self.c4
        self.fc = nn.Linear(fc_in_ch, num_class)

        self.init_weight()

    def init_weight(self):
        """Follows the weight initialization from the original SGN codebase."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        if 'spa' in self.gcn_list:
            for p_name, param in self.gcn_spatial.named_parameters():
                if f'w1.block.conv.conv.weight' in p_name:
                    init_0(param)
        if 'tem' in self.gcn_list:
            for p_name, param in self.gcn_temporal.named_parameters():
                if f'w1.block.conv.conv.weight' in p_name:
                    init_0(param)

    def forward(self, x: Tensor) -> Tuple[Tensor, list]:
        """Model forward pass.

        Args:
            x (Tensor): 3D joint in_pos of a sequence of skeletons.

        Returns:
            y (Tensor): logits.
            g (Tensor): Attention matrix for GCN.
        """
        assert x.shape[-1] % 3 == 0, "Only support input of xyz only."

        # Dynamic Representation -----------------------------------------------
        x = self.feature_extractor(x)
        assert x is not None

        # Joint and frame embeddings -------------------------------------------
        spa1, tem1 = self.semantic_embedding(x)

        # Joint-level Module ---------------------------------------------------
        # xyz embeddings
        if spa1 is None:
            x = x  # n,c,v,t
        else:
            # xyz emb (c) spa emb
            # xyz emb + spa emb
            x = fuse(x, spa1, self.sem_pos_fusion)

        if hasattr(self, 'x_emb_projection'):
            x = self.x_emb_projection(x)

        # temporal fusion pre gcn
        if self.sem_fra > 0 and self.sem_fra_location == 1:
            x = x + tem1

        # GCN ------------------------------------------------------------------
        s = x.shape

        if 'dual' in self.gcn_list:
            x_list, g_list = self.gcn_dual(x)
        else:
            x_list, g_list = [], []
            if hasattr(self, 'gcn_spatial'):
                x_spa, g_spa = self.gcn_spatial(x)
                x_list.append(x_spa)
                g_list.append(g_spa)
            if hasattr(self, 'gcn_temporal'):
                if self.gcn_tem == 0:
                    # swap axis
                    x_tem = x.transpose(-1, -2)
                    x_tem, g_tem = self.gcn_temporal(x_tem)
                    x_tem = x_tem.transpose(-1, -2)
                elif self.gcn_tem == 1:
                    # merge axis
                    x_tem = x.reshape((s[0], -1, s[-1], 1))
                    x_tem, g_tem = self.gcn_temporal(x_tem)
                    x_tem = x_tem.reshape((s[0], -1, s[2], s[3]))
                x_list.append(x_tem)
                g_list.append(g_tem)

        # Frame-level Module ---------------------------------------------------
        # spatial fusion post gcn
        if len(self.gcn_list) == 0:
            x = x
        elif len(self.gcn_list) == 1 or 'dual' in self.gcn_list:
            x = x_list[0]
        elif len(self.gcn_list) == 2:
            x = fuse(*x_list, self.gcn_fusion)
        else:
            raise ValueError("too many gcn definitions")

        # temporal fusion post gcn
        if self.sem_fra > 0 and self.sem_fra_location == 0:
            x = x + tem1

        # spatial pooling
        x = self.smp(x)

        # temporal MLP
        x = self.tem_mlp(x)

        # temporal pooling
        y = self.tmp(x)

        # Classification -------------------------------------------------------
        y = torch.flatten(y, 1)
        y = self.fc_dropout(y)
        y = self.fc(y)

        return y, g_list


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
                 dilations: List[int] = [1, 1],
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
            if normalizations[i] is None:
                norm_func = None
            else:
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
                         dilation=dilations[i],
                         bias=biases[i],
                         activation=activations[i],
                         normalization=norm_func,
                         dropout=dropouts[i],
                         prenorm=prenorm,
                         deterministic=False if dilations[i] > 1 else True)
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

    def forward(self, x: Tensor, x_n: Optional[Tensor] = None) -> Tensor:
        # x: n,c,v,t ; v=1 due to SMP
        x0 = x if x_n is None else x_n
        x = self.pool(x)
        for i in range(self.num_layers):
            x = getattr(self, f'cnn{i+1}')(x) + getattr(self, f'res{i+1}')(x)
        x += self.res(x0)
        return x


class MLPTemporalBranch(PyTorchModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: int = 0,
                 dropout: Optional[Type[PyTorchModule]] = None,
                 activation: Optional[Type[PyTorchModule]] = None,
                 normalization: Optional[Type[PyTorchModule]] = None,
                 prenorm: bool = False,

                 g_activation_fn: Optional[Type[PyTorchModule]] = None,

                 aspp_rates: Optional[List[int]] = None,

                 t_mode: int = 0,
                 t_kernel: int = 3,
                 t_maxpool_kwargs: Optional[dict] = None,
                 t_gcn_kwargs: Optional[dict] = None

                 ):
        super(MLPTemporalBranch, self).__init__()
        # aspp
        if aspp_rates is None or len(aspp_rates) == 0:
            self.aspp = nn.Identity()
        else:
            self.aspp = ASPP(in_channels,
                             in_channels,
                             bias=bias,
                             dilation=aspp_rates,
                             dropout=dropout,
                             activation=activation,
                             normalization=normalization)

        self.t_mode = t_mode

        # skip
        if t_mode == 0:
            self.cnn = nn.Identity()
        elif t_mode in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            # original sgn
            if t_mode == 1:
                idx = 2
                channels = [in_channels, in_channels, out_channels]
                kernel_sizes = [t_kernel, 1]
                paddings = [t_kernel//2, 0]
                residuals = [0 for _ in range(idx)]
                dropouts = [dropout, None]
            # original sgn with residual
            elif t_mode == 2:
                idx = 2
                channels = [in_channels, in_channels, out_channels]
                kernel_sizes = [t_kernel, 1]
                paddings = [t_kernel//2, 0]
                residuals = [1 for _ in range(idx)]
                dropouts = [dropout, None]
            # all 3x3
            elif t_mode == 3:
                idx = 2
                channels = [in_channels, in_channels, out_channels]
                kernel_sizes = [t_kernel for _ in range(2)]
                paddings = [t_kernel//2 for _ in range(2)]
                residuals = [0 for _ in range(2)]
                dropouts = [dropout, None]
            # all 3x3 with residual
            elif t_mode == 4:
                idx = 2
                channels = [in_channels, in_channels, out_channels]
                kernel_sizes = [t_kernel for _ in range(2)]
                paddings = [t_kernel//2 for _ in range(2)]
                residuals = [1 for _ in range(2)]
                dropouts = [dropout, None]
            # 3 layers
            elif t_mode == 5:
                idx = 3
                channels = [in_channels, in_channels,
                            in_channels, out_channels]
                kernel_sizes = [t_kernel, 1, 1]
                paddings = [t_kernel//2, 0, 0]
                residuals = [0 for _ in range(3)]
                dropouts = [dropout, None, None]
            # 3 layers with residual
            elif t_mode == 6:
                idx = 3
                channels = [in_channels, in_channels,
                            in_channels, out_channels]
                kernel_sizes = [t_kernel, 1, 1]
                paddings = [t_kernel//2, 0, 0]
                residuals = [1 for _ in range(3)]
                dropouts = [dropout, None, None]
            # original sgn + all dropout
            elif t_mode == 7:
                idx = 2
                channels = [in_channels, in_channels, out_channels]
                kernel_sizes = [t_kernel, 1]
                paddings = [t_kernel//2, 0]
                residuals = [0 for _ in range(idx)]
                dropouts = [dropout, dropout]
            # original sgn + all dropout + residual
            elif t_mode == 8:
                idx = 2
                channels = [in_channels, in_channels, out_channels]
                kernel_sizes = [t_kernel, 1]
                paddings = [t_kernel//2, 0]
                residuals = [1 for _ in range(idx)]
                dropouts = [dropout, dropout]
            # original sgn with quarter input channel
            elif t_mode == 9:
                idx = 2
                channels = [in_channels, in_channels//4, out_channels]
                kernel_sizes = [t_kernel, 1]
                paddings = [t_kernel//2, 0]
                residuals = [0 for _ in range(idx)]
                dropouts = [dropout, None]
            # original sgn with quarter input channel + residual
            elif t_mode == 10:
                idx = 2
                channels = [in_channels, in_channels//4, out_channels]
                kernel_sizes = [t_kernel, 1]
                paddings = [t_kernel//2, 0]
                residuals = [1 for _ in range(idx)]
                dropouts = [dropout, None]
            else:
                raise ValueError("Unknown t_mode...")
            self.cnn = MLPTemporal(
                channels=channels,
                kernel_sizes=kernel_sizes,
                paddings=paddings,
                biases=[bias for _ in range(idx)],
                residuals=residuals,
                dropouts=dropouts,
                activations=[activation for _ in range(idx)],
                normalizations=[normalization for _ in range(idx)],
                maxpool_kwargs=t_maxpool_kwargs,
                prenorm=prenorm
            )
        # gcn only
        elif t_mode == 100:
            assert t_gcn_kwargs is not None
            t_gcn_kwargs['gcn_dims'] = [in_channels] + t_gcn_kwargs['gcn_dims']
            self.cnn = GCNSpatialBlock(
                0,
                0,
                bias=bias,
                activation=activation,
                normalization=normalization,
                g_activation=g_activation_fn,
                **t_gcn_kwargs
            )
        # gcn + mlp
        elif t_mode in [101, 102]:
            assert t_gcn_kwargs is not None
            t_gcn_kwargs['gcn_dims'] = [in_channels] + t_gcn_kwargs['gcn_dims']
            if t_mode == 101:
                idx = 2
                channels = [in_channels, in_channels, out_channels]
                kernel_sizes = [t_kernel, 1]
                paddings = [t_kernel//2, 0]
                residuals = [0 for _ in range(idx)]
                dropouts = [dropout, None]
            elif t_mode == 102:
                idx = 2
                channels = [in_channels, in_channels, out_channels]
                kernel_sizes = [t_kernel, 1]
                paddings = [t_kernel//2, 0]
                residuals = [1 for _ in range(idx)]
                dropouts = [dropout, None]
            self.cnn = nn.Sequential(OrderedDict([
                ('GCN',
                    GCNSpatialBlock(
                        0,
                        0,
                        bias=bias,
                        activation=activation,
                        normalization=normalization,
                        g_activation=g_activation_fn,
                        return_g=False,
                        **t_gcn_kwargs
                    )),
                ('MLP',
                    MLPTemporal(
                        channels=channels,
                        kernel_sizes=kernel_sizes,
                        paddings=paddings,
                        biases=[bias for _ in range(idx)],
                        residuals=residuals,
                        dropouts=dropouts,
                        activations=[activation for _ in range(idx)],
                        normalizations=[normalization for _ in range(idx)],
                        maxpool_kwargs=t_maxpool_kwargs,
                        prenorm=prenorm
                    ))
            ]))
        else:
            raise ValueError('Unknown t_mode')

    def forward(self, x: Tensor) -> Tensor:
        # x: n,c,v,t ; v=1 due to SMP
        x = self.aspp(x)
        if self.t_mode == 100:
            x, _ = self.cnn(x.transpose(-1, -2))
            x = x.transpose(-1, -2)
        elif self.t_mode in [101, 102]:
            x = self.cnn.GCN(x.transpose(-1, -2))
            x = self.cnn.MLP(x.transpose(-1, -2))
        else:
            x = self.cnn(x)
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
                 g_proj_shared: bool = False,
                 ):
        super(GCNSpatialG, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=bias,
                                          activation=activation,
                                          normalization=normalization)
        self.g1 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                       kernel_size=self.kernel_size, padding=self.padding)
        if g_proj_shared:
            self.g2 = self.g1
        else:
            self.g2 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                           kernel_size=self.kernel_size, padding=self.padding)
        self.act = self.activation(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
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
        if not self.prenorm:
            self.norm = self.normalization(self.out_channels)
        self.act = self.activation()
        self.drop = nn.Identity() if self.dropout is None else self.dropout()
        self.w1 = Conv(self.in_channels, self.out_channels, bias=self.bias)
        self.w2 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                       kernel_size=self.kernel_size, padding=self.padding)

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
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
                 gcn_dims: List[int] = [128, 256, 256],
                 gcn_residual: T3 = [0, 0, 0],
                 gcn_prenorm: bool = False,
                 g_proj_dim: T3 = 256,
                 g_kernel: int = 1,
                 g_proj_shared: bool = False,
                 g_activation: T1 = nn.Softmax,
                 ffn_mode: int = 0,
                 ffn_prenorm: int = 0,
                 return_g: bool = True,
                 segments: int = 20,
                 ):
        super(GCNSpatialBlock, self).__init__(*args,
                                              kernel_size=kernel_size,
                                              padding=padding,
                                              bias=bias,
                                              dropout=dropout,
                                              activation=activation,
                                              normalization=normalization)
        self.ffn_mode = ffn_mode
        self.return_g = return_g
        self.num_blocks = len(gcn_dims) - 1
        self.g_shared = isinstance(g_proj_dim, int)
        if self.g_shared:
            self.gcn_g1 = GCNSpatialG(gcn_dims[0],
                                      g_proj_dim,
                                      bias=self.bias,
                                      kernel_size=g_kernel,
                                      padding=g_kernel//2,
                                      activation=g_activation,
                                      normalization=self.normalization,
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
                                   prenorm=gcn_prenorm))

        if gcn_prenorm:
            for i in range(self.num_blocks):
                setattr(self, f'gcn_prenorm{i+1}',
                        self.normalization(gcn_dims[i]))

        if isinstance(gcn_residual, list):
            assert len(gcn_residual) == self.num_blocks
            for i, r in enumerate(gcn_residual):
                if r == 0:
                    setattr(self, f'gcn_res{i+1}', null_fn)
                elif r == 1:
                    if gcn_dims[i] == gcn_dims[i+1]:
                        setattr(self, f'gcn_res{i+1}', nn.Identity())
                    else:
                        setattr(self, f'gcn_res{i+1}', Conv(gcn_dims[i],
                                                            gcn_dims[i+1],
                                                            bias=self.bias))
                else:
                    raise ValueError("Unknown residual modes...")
            self.res = null_fn

        elif isinstance(gcn_residual, int):
            if gcn_residual == 0:
                self.res = null_fn
            elif gcn_residual == 1:
                if gcn_dims[0] == gcn_dims[-1]:
                    self.res = nn.Identity()
                else:
                    self.res = Conv(gcn_dims[0], gcn_dims[-1], bias=self.bias)
            else:
                raise ValueError("Unknown residual modes...")

        else:
            raise ValueError("Unknown residual modes...")

        if ffn_mode == 201:
            self.ffn_smp = nn.AdaptiveMaxPool2d((segments, 1))
            self.ffn_gcn_g1 = GCNSpatialG(gcn_dims[1],
                                          g_proj_dim,
                                          bias=self.bias,
                                          kernel_size=g_kernel,
                                          padding=g_kernel//2,
                                          activation=g_activation,
                                          normalization=self.normalization,
                                          g_proj_shared=g_proj_shared)

            for i in range(self.num_blocks):
                setattr(self,
                        f'ffn_gcn{i+1}',
                        GCNSpatialUnit(gcn_dims[i+1],
                                       gcn_dims[i+1],
                                       bias=self.bias,
                                       kernel_size=self.kernel_size,
                                       padding=self.padding,
                                       dropout=self.dropout,
                                       activation=self.activation,
                                       normalization=self.normalization,
                                       prenorm=gcn_prenorm))

            if gcn_prenorm:
                for i in range(self.num_blocks):
                    setattr(self,
                            f'ffn_gcn_prenorm{i+1}',
                            self.normalization(gcn_dims[i]))

            assert len(gcn_residual) == self.num_blocks
            for i, r in enumerate(gcn_residual):
                if r == 0:
                    setattr(self, f'ffn_gcn_res{i+1}', null_fn)
                elif r == 1:
                    setattr(self, f'ffn_gcn_res{i+1}', nn.Identity())
                else:
                    raise ValueError("Unknown residual modes...")

        elif ffn_mode > 100:
            for i in range(self.num_blocks):
                if ffn_mode == 101:
                    dilation = [0, 1, 3]
                elif ffn_mode == 102:
                    dilation = [1, 3, 5]
                elif ffn_mode == 103:
                    dilation = [3, 5, 7]
                elif ffn_mode == 104:
                    dilation = [3, 7, 11]
                setattr(self,
                        f'ffn{i+1}',
                        ASPP(gcn_dims[i+1],
                             gcn_dims[i+1],
                             bias=self.bias,
                             dilation=dilation,
                             dropout=self.dropout,
                             activation=self.activation,
                             normalization=self.normalization,
                             residual=1))
                if ffn_prenorm:
                    setattr(self, f'ffn_prenorm{i+1}',
                            self.normalization(gcn_dims[i+1]))
        elif ffn_mode > 0:
            for i in range(self.num_blocks):
                if ffn_mode == 1:
                    # transformer style, residual
                    channels = [gcn_dims[i+1], gcn_dims[i+1]*4, gcn_dims[i+1]]
                    kernel_sizes = [1, 1]
                    paddings = [0, 0]
                    dilations = [1, 1]
                    biases = [self.bias, self.bias]
                    residuals = [0, 0]
                    dropouts = [self.dropout, self.dropout]
                    activations = [self.activation, None]
                    normalizations = [None, None]
                    residual = 1
                elif ffn_mode == 2:
                    # bottleneck 1x3, postnorm
                    channels = [gcn_dims[i+1], gcn_dims[i+1]//4, gcn_dims[i+1]]
                    kernel_sizes = [3, 3]
                    paddings = [1, 1]
                    dilations = [1, 1]
                    biases = [self.bias, self.bias]
                    residuals = [0, 0]
                    dropouts = [self.dropout, self.dropout]
                    activations = [self.activation, self.activation]
                    normalizations = [self.normalization, self.normalization]
                    residual = 1
                elif ffn_mode == 3:
                    # dilation [3 7 11] + 1x1proj, norm, residual
                    channels = [gcn_dims[i+1], gcn_dims[i+1], gcn_dims[i+1]]
                    kernel_sizes = [3, 1]
                    paddings = [3+(i*4), 0]
                    dilations = [3+(i*4), 1]
                    biases = [self.bias, self.bias]
                    residuals = [0, 0]
                    dropouts = [self.dropout, self.dropout]
                    activations = [self.activation, self.activation]
                    normalizations = [self.normalization, self.normalization]
                    residual = 1
                setattr(self,
                        f'ffn{i+1}',
                        MLPTemporal(
                            channels=channels,
                            kernel_sizes=kernel_sizes,
                            paddings=paddings,
                            dilations=dilations,
                            biases=biases,
                            residuals=residuals,
                            dropouts=dropouts,
                            activations=activations,
                            normalizations=normalizations,
                            maxpool_kwargs=None,
                            residual=residual,
                            prenorm=ffn_prenorm)
                        )
                if ffn_prenorm:
                    setattr(self, f'ffn_prenorm{i+1}',
                            self.normalization(channels[0]))
        else:
            for i in range(self.num_blocks):
                setattr(self, f'ffn{i+1}', Identity())

    def forward(self, x: Tensor) -> Tensor:
        x0 = x
        g = []
        ffn_g = []
        for i in range(self.num_blocks):
            if hasattr(self, f'gcn_prenorm{i+1}'):
                x1 = getattr(self, f'gcn_prenorm{i+1}')(x)
            else:
                x1 = x
            if (self.g_shared and len(g) == 0) or not self.g_shared:
                g1 = getattr(self, f'gcn_g{i+1}')(x1)
                g.append(g1)
            x = getattr(self, f'gcn{i+1}')(x1, g1) + \
                getattr(self, f'gcn_res{i+1}')(x)

            if self.ffn_mode == 201:
                x = x.transpose(-1, -2)  # nctv
                if hasattr(self, f'ffn_gcn_prenorm{i+1}'):
                    x1 = getattr(self, f'ffn_gcn_prenorm{i+1}')(x)
                else:
                    x1 = x
                x2 = self.ffn_smp(x1)  # nc1t
                if (self.g_shared and len(ffn_g) == 0) or not self.g_shared:
                    ffn_g1 = getattr(self, f'ffn_gcn_g{i+1}')(x2)
                    ffn_g.append(ffn_g1)
                x = getattr(self, f'ffn_gcn{i+1}')(x1, ffn_g1) + \
                    getattr(self, f'ffn_gcn_res{i+1}')(x)
                x = x.transpose(-1, -2)  # nct1

            else:
                if hasattr(self, f'ffn_prenorm{i+1}'):
                    x1 = getattr(self, f'ffn_prenorm{i+1}')(x)
                else:
                    x1 = x
                try:
                    x = getattr(self, f'ffn{i+1}')(x1, x)
                except TypeError:
                    x = getattr(self, f'ffn{i+1}')(x1)
                except Exception:
                    raise ValueError("Missing ffn init or wrong inputs.")

        x += self.res(x0)

        if self.return_g:
            return x, g+ffn_g
        else:
            return x


class DualGCNSpatialBlock(PyTorchModule):
    def __init__(self, gcn1: Module, gcn2: Module, mode=1):
        super(DualGCNSpatialBlock, self).__init__()
        self.mode = 1
        assert mode == 1
        self.gcn1 = gcn1
        self.gcn2 = gcn2
        assert gcn1 is not None and gcn2 is not None
        assert gcn1.num_blocks == gcn2.num_blocks

    def forward(self, x: Tensor) -> Tensor:
        # x: n,c,v,t
        x0 = x
        x_list, g_list = [], []
        gcn_g1, gcn_g2 = [], []
        for i in range(self.gcn1.num_blocks):
            # gcn1 ----
            if self.gcn1.prenorm:
                x1 = getattr(self.gcn1, f'gcn_prenorm{i+1}')(x)
            else:
                x1 = x
            if (self.gcn1.g_shared and len(gcn_g1) == 0) or not self.gcn1.g_shared:  # noqa
                g1 = getattr(self.gcn1, f'gcn_g{i+1}')(x1)
                gcn_g1.append(g1)
            x1 = getattr(self.gcn1, f'gcn{i+1}')(x1, g1) + \
                getattr(self.gcn1, f'gcn_res{i+1}')(x)
            if self.gcn1.prenorm:
                x2 = getattr(self.gcn1, f'ffn_prenorm{i+1}')(x1)
            else:
                x2 = x1
            gcn1_x = getattr(self.gcn1, f'ffn{i+1}')(x2, x1)

            # gcn2 -----
            x = x.transpose(-1, -2)
            if self.gcn2.prenorm:
                x1 = getattr(self.gcn2, f'gcn_prenorm{i+1}')(x)
            else:
                x1 = x
            if (self.gcn2.g_shared and len(gcn_g2) == 0) or not self.gcn2.g_shared:  # noqa
                g2 = getattr(self.gcn2, f'gcn_g{i+1}')(x1)
                gcn_g2.append(g2)
            x1 = getattr(self.gcn2, f'gcn{i+1}')(x1, g2) + \
                getattr(self.gcn2, f'gcn_res{i+1}')(x)
            if self.gcn2.prenorm:
                x2 = getattr(self.gcn2, f'ffn_prenorm{i+1}')(x1)
            else:
                x2 = x1
            gcn2_x = getattr(self.gcn2, f'ffn{i+1}')(x2, x1)
            gcn2_x = gcn2_x.transpose(-1, -2)

            # aggregation ---
            if self.mode == 1:
                x = gcn1_x + gcn2_x

        x += self.gcn1.res(x0)
        x_list.append(x)
        g_list.append([gcn_g1, gcn_g2])
        return x_list, g_list


class FeatureExtractor(PyTorchModule):
    def __init__(self,
                 in_pos: int,
                 in_vel: int,
                 in_pos_kwargs: dict,
                 in_vel_kwargs: dict,
                 ):
        super(FeatureExtractor, self).__init__()
        self.in_pos = in_pos
        self.in_vel = in_vel
        if self.in_pos > 0:
            self.pos_embed = Embedding(**in_pos_kwargs)
        if self.in_vel > 0:
            self.vel_embed = Embedding(**in_vel_kwargs)
        if self.in_pos == 0 and self.in_vel == 0:
            raise ValueError("Input args are faulty...")

    def forward(self, x: Tensor) -> Optional[Tensor]:
        bs, step, dim = x.shape
        num_point = dim // 3
        x1 = x.view((bs, step, num_point, 3))  # n,t,v,c
        x = x1.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
        dif = x[:, :, :, 1:] - x[:, :, :, 0:-1]  # n,c,v,t-1
        dif = pad_zeros(dif)
        if self.in_pos > 0 and self.in_vel > 0:
            pos = self.pos_embed(x)
            dif = self.vel_embed(dif)
            dy1 = pos + dif  # n,c,v,t
        elif self.in_pos > 0:
            dy1 = self.pos_embed(x)
        elif self.in_vel > 0:
            dy1 = self.vel_embed(dif)
        else:
            dy1 = None
        return dy1


class SemanticEmbedding(PyTorchModule):
    def __init__(self,
                 num_point: int,
                 num_segment: int,
                 sem_pos: int,
                 sem_fra: int,
                 spa_embed_kwargs: dict,
                 tem_embed_kwargs: dict,
                 ):
        super(SemanticEmbedding, self).__init__()
        self.num_point = num_point
        self.num_segment = num_segment
        self.sem_pos = sem_pos
        self.sem_fra = sem_fra
        # Joint Embedding
        if self.sem_pos > 0:
            self.spa = OneHotTensor(self.num_point, self.num_segment, mode=0)
            self.spa_embed = Embedding(**spa_embed_kwargs)
        # Frame Embedding
        if self.sem_fra > 0:
            self.tem = OneHotTensor(self.num_segment, self.num_point, mode=1)
            self.tem_embed = Embedding(**tem_embed_kwargs)

    def forward(self, x: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        spa1, tem1 = None, None
        if self.sem_pos > 0:
            spa1 = self.spa_embed(self.spa(x.shape[0]))
        if self.sem_fra > 0:
            tem1 = self.tem_embed(self.tem(x.shape[0]))
        return spa1, tem1


if __name__ == '__main__':

    batch_size = 64

    inputs = torch.ones(batch_size, 20, 75)
    subjects = torch.ones(batch_size, 20, 1)

    model = SGN(num_segment=20,
                # c_multiplier=[0.25, 0.25, 0.25, 0.25],
                # gcn_spa_dims=[c2*0.25, c3*0.25, c3*0.25],
                # sem_pos_fusion=1,
                # sem_fra_fusion=1,
                # sem_fra_location=0,
                # x_emb_proj=2,
                # gcn_list=['spa', 'tem', 'dual'],
                gcn_list=['spa'],
                # gcn_fusion=0,
                gcn_spa_g_kernel=1,
                gcn_spa_g_proj_shared=False,
                gcn_spa_g_proj_dim=256,
                gcn_spa_t_kernel=1,
                gcn_spa_dropout=0.0,
                gcn_spa_gcn_residual=[0, 0, 0],
                gcn_spa_dims=[128, 256, 256],
                gcn_spa_prenorm=False,
                gcn_spa_ffn_prenorm=False,
                gcn_spa_ffn=201,
                # gcn_tem=0,
                # gcn_tem_dims=[c2*25, c3*25, c3*25],
                # t_mode=1,
                # t_gcn_dims=[256, 256, 256]
                # spatial_maxpool=1,
                # temporal_maxpool=0,

                )
    model(inputs)
    # print(model)

    try:
        flops = FlopCountAnalysis(model, inputs)
        print(flops.total())
        # print(flops.by_module_and_operator())
        # print(flop_count_table(flops))
    except NameError:
        print("Warning: fvcore is not found")
