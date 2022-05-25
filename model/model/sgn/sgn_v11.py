# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Taken from :
# https://github.com/microsoft/SGN/blob/master/model.py

# Code refractored

# Continue from on sgn_v10
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

import inspect
import math
from typing import OrderedDict, Tuple, Optional, Union, Type, List, Any

from model.module import *
from model.module.layernorm import LayerNorm
from model.resource.common_ntu import *

from utils.utils import *


T1 = Type[PyTorchModule]
T2 = List[Optional[Type[PyTorchModule]]]
T3 = Union[List[int], int]

EMB_MODES = [1, 2, 3, 4, 5, 6, 7, 8]
T_MODES = [0, 1, 2]


def null_fn(x: Any) -> int:
    return 0


def init_0(x: Tensor):
    return nn.init.constant_(x, 0)


def pad_zeros(x: Tensor) -> Tensor:
    return torch.cat([x.new(*x.shape[:-1], 1).zero_(), x], dim=-1)


def residual_layer(residual: int, in_ch: int, out_ch: int, bias: int = 0):
    if residual == 0:
        return null_fn
    elif residual == 1:
        if in_ch == out_ch:
            return nn.Identity()
        else:
            return Conv(in_ch, out_ch, bias=bias)
    else:
        raise ValueError("Unknown residual modes...")


def fuse_features(x1: Tensor, x2: Tensor, mode: int) -> Tensor:
    if mode == 0:
        return torch.cat([x1, x2], 1)
    elif mode == 1:
        return x1 + x2
    else:
        raise ValueError('Unknown feature fusion arg')


def activation_fn(act_type: str) -> Type[Type[PyTorchModule]]:
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
    ffn_mode = [0, 1, 2, 3, 101, 102, 103, 104, 201, 202]
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

                 xem_projection: int = 0,
                 input_position: int = 1,
                 input_velocity: int = 1,
                 semantic_joint: int = 1,
                 semantic_frame: int = 1,

                 semantic_joint_fusion: int = 0,
                 semantic_frame_fusion: int = 1,
                 semantic_frame_location: int = 0,

                 sgcn_dims: Optional[list] = None,  # [c2, c3, c3],
                 sgcn_kernel: int = 1,  # residual connection in GCN
                 sgcn_padding: int = 0,  # residual connection in GCN
                 sgcn_dropout: float = 0.0,  # residual connection in GCN
                 # int for global res, list for individual gcn
                 sgcn_residual: T3 = [0, 0, 0],
                 sgcn_prenorm: bool = False,
                 sgcn_g_kernel: int = 1,
                 sgcn_g_proj_dim: Optional[T3] = None,  # c3
                 sgcn_g_proj_shared: bool = False,

                 gcn_fpn: int = -1,

                 spatial_maxpool: int = 1,
                 temporal_maxpool: int = 1,

                 # ORDER IS IMPORTANT
                 multi_t: List[List[int]] = [[], [], [3]],
                 multi_t_shared: int = 0,

                 t_mode: int = 1,
                 t_maxpool_kwargs: Optional[dict] = None,
                 aspp_rates: Optional[list] = None,

                 ):
        super(SGN, self).__init__()

        # Base SGN args --------------------------------------------------------
        self.num_class = num_class
        self.num_point = num_point
        self.num_segment = num_segment
        self.in_channels = in_channels
        self.bias = bias
        self.dropout_fn = lambda: nn.Dropout2d(dropout2d)
        self.fc_dropout = nn.Dropout(dropout)

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

        # Input + Semantics ----------------------------------------------------
        self.input_position = input_position
        self.input_velocity = input_velocity
        self.semantic_joint = semantic_joint
        self.semantic_frame = semantic_frame
        self.xem_projection = xem_projection  # projection layer pre GCN
        assert self.input_position in self.emb_modes
        assert self.input_velocity in self.emb_modes
        assert self.semantic_joint in self.emb_modes
        assert self.semantic_frame in self.emb_modes
        assert self.xem_projection in self.emb_modes
        if self.input_position == 0 and self.semantic_joint > 0:
            raise ValueError("input_position is 0 but semantic_joint is not")

        # 0: concat, 1: sum
        self.semantic_joint_fusion = semantic_joint_fusion
        # 0: concat, 1: sum
        self.semantic_frame_fusion = semantic_frame_fusion
        # 0 = add after GCN, 1 = add before GCN
        self.semantic_frame_location = semantic_frame_location
        assert self.semantic_frame_location in [0, 1]

        # Dynamic Representation -----------------------------------------------
        self.feature_extractor = FeatureExtractor(
            in_pos=self.input_position,
            in_vel=self.input_velocity,
            in_pos_emb_kwargs=dict(
                in_channels=self.in_channels,
                out_channels=self.c1,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                in_norm=self.normalization_fn_1d,
                num_point=self.num_point,
            ),
            in_vel_emb_kwargs=dict(
                in_channels=self.in_channels,
                out_channels=self.c1,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                in_norm=self.normalization_fn_1d,
                num_point=self.num_point,
            )
        )

        # Joint and frame embeddings -------------------------------------------

        # Input dim to the GCN
        if self.semantic_joint == 0:
            self.gcn_in_ch = self.c1
        else:
            if self.semantic_joint_fusion == 0:
                self.gcn_in_ch = self.c1 * 2
            elif self.semantic_joint_fusion == 1:
                self.gcn_in_ch = self.c1

        # post gcn
        if self.semantic_frame_location == 0:
            if gcn_fpn == 2:
                out_channels = self.gcn_in_ch
            else:
                out_channels = self.c3
        # pre gcn
        elif self.semantic_frame_location == 1:
            out_channels = self.gcn_in_ch

        self.semantic_embedding = SemanticEmbedding(
            num_point=self.num_point,
            num_segment=self.num_segment,
            sem_spa=self.semantic_joint,
            sem_tem=self.semantic_frame,
            sem_spa_emb_kwargs=dict(
                in_channels=self.num_point,
                out_channels=self.c1,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                num_point=self.num_point,
                mode=self.semantic_joint
            ),
            sem_tem_emb_kwargs=dict(
                in_channels=self.num_segment,
                out_channels=out_channels,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                num_point=self.num_point,
                mode=self.semantic_frame
            )
        )

        # Spatial GCN ----------------------------------------------------------

        # projection layer pre GCN
        if self.xem_projection > 0:
            self.x_emb_projection = Embedding(
                in_channels=self.gcn_in_ch,
                out_channels=self.gcn_in_ch,
                bias=self.bias,
                # dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                num_point=self.num_point,
                mode=self.x_emb_proj
            )

        if sgcn_dims is None:
            sgcn_dims = [self.c2, self.c3, self.c3]
        if sgcn_g_proj_dim is None:
            sgcn_g_proj_dim = self.c3
        self.sgcn_dims = sgcn_dims
        self.sgcn = GCNSpatialBlock(
            kernel_size=sgcn_kernel,
            padding=sgcn_padding,
            bias=self.bias,
            dropout=lambda: nn.Dropout2d(sgcn_dropout),
            activation=self.activation_fn,
            normalization=self.normalization_fn,
            gcn_dims=[self.gcn_in_ch] + sgcn_dims,
            gcn_residual=sgcn_residual,
            gcn_prenorm=sgcn_prenorm,
            g_kernel=sgcn_g_kernel,
            g_proj_dim=sgcn_g_proj_dim,
            g_proj_shared=sgcn_g_proj_shared,
            g_activation=self.g_activation_fn,
            return_g=True,
            return_gcn_list=True,
        )

        # GCN FPN --------------------------------------------------------------
        # 0 no fpn
        # 1 proj and sum
        # 2 proj to lower
        # 3 proj
        # 4 proj and concat
        self.gcn_fpn = gcn_fpn
        if self.gcn_fpn == 0:
            assert self.semantic_frame_location == 1
        elif self.gcn_fpn in [1, 3, 4]:
            for i in range(len(sgcn_dims)):
                setattr(self,
                        f'fpn_proj{i+1}',
                        Conv(sgcn_dims[i],
                             sgcn_dims[-1],
                             bias=self.bias,
                             activation=self.activation_fn,
                             normalization=lambda: self.normalization_fn(
                                 sgcn_dims[-1]),
                             #  dropout=self.dropout_fn,
                             ))
        elif self.gcn_fpn == 2:
            for i in range(len(sgcn_dims)):
                setattr(self,
                        f'fpn_proj{i+1}',
                        Conv(sgcn_dims[i],
                             sgcn_dims[0],
                             bias=self.bias,
                             activation=self.activation_fn,
                             normalization=lambda: self.normalization_fn(
                                 sgcn_dims[0]),
                             #  dropout=self.dropout_fn,
                             ))

        # Frame level module ---------------------------------------------------
        self.t_mode = t_mode
        self.t_maxpool_kwargs = t_maxpool_kwargs
        self.aspp_rates = aspp_rates  # dilation rates
        self.multi_t = multi_t  # list of list : gcn layers -> kernel sizes
        assert len(self.multi_t) == len(sgcn_dims)
        # 0: no
        # 1: intra gcn share (no sense)
        # 2: inter gcn share (between layer share)
        self.multi_t_shared = multi_t_shared
        assert self.multi_t_shared in [0, 2]

        # loop through the gcn fpn
        for i, (sgcn_dim, t_kernels) in enumerate(zip(self.sgcn_dims,
                                                      self.multi_t)):

            # # all the kernel size for that gcn layer should be the same
            # if self.multi_t_shared == 1 and len(t_kernels) > 0:
            #     assert len(set(t_kernels)) == 1

            for j, t_kernel in enumerate(t_kernels):

                if self.gcn_fpn == 0:
                    in_ch = sgcn_dim
                elif self.gcn_fpn in [1, 3]:
                    in_ch = sgcn_dims[-1]
                elif self.gcn_fpn == 2:
                    in_ch = sgcn_dims[0]
                elif self.gcn_fpn == 4:
                    in_ch = sgcn_dims[-1]*3
                else:
                    in_ch = sgcn_dims[-1]

                name = f'tem_mlp_{i+1}_{j+1}_k{t_kernel}'

                if self.multi_t_shared == 2:
                    cont = False
                    for k in range(i+1):
                        if getattr(self,
                                   f'tem_mlp_{k+1}_{j+1}_k{t_kernel}',
                                   None) is not None:
                            cont = True
                    if cont:
                        continue

                setattr(self,
                        name,
                        MLPTemporalBranch(
                            in_channels=in_ch,
                            out_channels=self.c4,
                            kernel_size=t_kernel,
                            bias=self.bias,
                            dropout=self.dropout_fn,
                            activation=self.activation_fn,
                            normalization=self.normalization_fn,
                            prenorm=self.prenorm,
                            t_mode=self.t_mode,
                            aspp_rates=self.aspp_rates,
                            maxpool_kwargs=self.t_maxpool_kwargs,
                        ))

                # if self.multi_t_shared == 1:
                #     break

        # Maxpool --------------------------------------------------------------
        # 0 no pool, 1 pool, 2 projection, 3 no pool but merge channels
        self.spatial_maxpool = spatial_maxpool
        self.temporal_maxpool = temporal_maxpool
        assert self.spatial_maxpool in [0, 1]
        assert self.temporal_maxpool in [0, 1]

        # 0=nn.Identity, 1=pool, 2=lerge kernel, 3= n,cv,1,t
        if self.spatial_maxpool == 0:
            self.smp = nn.Identity()
        elif self.spatial_maxpool == 1:
            self.smp = nn.AdaptiveMaxPool2d((1, self.num_segment))
        else:
            raise ValueError("Unknown spatial_maxpool")

        if self.temporal_maxpool == 0:
            self.tmp = nn.Identity()
        elif self.temporal_maxpool == 1:
            self.tmp = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError("Unknown temporal_maxpool")

        # Classifier ---------------------------------------------------------
        if self.t_mode == 0:
            fc_in_ch = self.c3
        elif self.spatial_maxpool == 0 and self.temporal_maxpool == 0:
            fc_in_ch = self.c4 * self.num_segment * self.num_point
        elif self.temporal_maxpool == 0:
            fc_in_ch = self.c4 * self.num_segment
        else:
            fc_in_ch = self.c4
        self.fc = nn.Linear(fc_in_ch, num_class)

        # Init weight ----------------------------------------------------------
        self.init_weight()

    def init_weight(self):
        """Follows the weight initialization from the original SGN codebase."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        for p_name, param in self.sgcn.named_parameters():
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
        spa_emb, tem_emb = self.semantic_embedding(x)

        # Joint-level Module ---------------------------------------------------
        # xyz embeddings
        if spa_emb is None:
            x = x  # n,c,v,t
        else:
            # xyz emb (c) spa emb
            # xyz emb + spa emb
            x = fuse_features(x, spa_emb, self.semantic_joint_fusion)

        if hasattr(self, 'x_emb_projection'):
            x = self.x_emb_projection(x)

        # temporal fusion pre gcn
        if self.semantic_frame > 0 and self.semantic_frame_location == 1:
            x = x + tem_emb

        # GCN ------------------------------------------------------------------
        _, g_spa, x_spa_list = self.sgcn(x)

        # gcn fpn
        if self.gcn_fpn == 0:
            x_list = x_spa_list
        elif self.gcn_fpn in [1, 2]:
            assert hasattr(self, 'sgcn')
            x_list = [getattr(self, f'fpn_proj{i+1}')(x_spa_list[i])
                      for i in range(len(x_spa_list))]
            x_list = [x_list[2] + x_list[1] + x_list[0],
                      x_list[2] + x_list[1],
                      x_list[2]]
        elif self.gcn_fpn in [3, 4]:
            assert hasattr(self, 'sgcn')
            x_list = [getattr(self, f'fpn_proj{i+1}')(x_spa_list[i])
                      for i in range(len(x_spa_list))]
        else:
            x_list = [None, None, x_spa_list[-1]]

        # Frame-level Module ---------------------------------------------------
        # temporal fusion post gcn
        if self.semantic_frame > 0 and self.semantic_frame_location == 0:
            x_list = [i + tem_emb if i is not None else None for i in x_list]

        # spatial pooling
        x_list = [self.smp(i) if i is not None else None for i in x_list]

        if self.gcn_fpn == 4:
            x_list = [None, None, torch.cat(x_list, dim=1)]

        # temporal MLP
        _x_list = []
        for i, t_kernels in enumerate(self.multi_t):
            for j, t_kernel in enumerate(t_kernels):

                if x_list[i] is None:
                    continue

                name = f'tem_mlp_{i+1}_{j+1}_k{t_kernel}'

                # if self.multi_t_shared == 1:
                #     name = f'tem_mlp_{i+1}_{1}_k{t_kernel}'
                if self.multi_t_shared == 2:
                    for k in range(i):
                        if getattr(self,
                                   f'tem_mlp_{k+1}_{j+1}_k{t_kernel}',
                                   None) is not None:
                            name = f'tem_mlp_{k+1}_{j+1}_k{t_kernel}'
                            break

                _x_list.append(getattr(self, name)(x_list[i]))

        x = torch.mean(torch.stack(_x_list, dim=0), dim=0)

        # temporal pooling
        y = self.tmp(x)

        # Classification -------------------------------------------------------
        y = torch.flatten(y, 1)
        y = self.fc_dropout(y)
        y = self.fc(y)

        return y, g_spa


class DataNorm(PyTorchModule):
    def __init__(self, dim: int, normalization: T1 = nn.BatchNorm1d):
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
                 #  inter_channels: list = [0],
                 #  residual: int = 0,
                 num_point: int = 25,
                 mode: int = 1
                 ):
        super(Embedding, self).__init__(in_channels,
                                        out_channels,
                                        bias=bias,
                                        dropout=dropout,
                                        activation=activation,
                                        normalization=normalization)
        assert mode in EMB_MODES
        self.mode = mode

        if in_norm is not None:
            self.norm = DataNorm(self.in_channels * num_point, in_norm)
        else:
            self.norm = nn.Identity()

        if self.mode // 100 == 0:
            if self.mode == 1:
                ch_list = [self.in_channels,
                           self.out_channels, self.out_channels]
                residual = 0
            elif self.mode == 2:
                ch_list = [self.in_channels,
                           self.out_channels, self.out_channels]
                residual = 1
            elif self.mode == 3:
                ch_list = [self.in_channels,
                           self.out_channels, self.out_channels,
                           self.out_channels]
                residual = 0
            elif self.mode == 4:
                ch_list = [self.in_channels,
                           self.out_channels, self.out_channels,
                           self.out_channels, self.out_channels]
                residual = 0

            self.num_layers = len(ch_list) - 1
            for i in range(self.num_layers):
                setattr(self, f'cnn{i+1}', Conv(ch_list[i],
                                                ch_list[i+1],
                                                bias=self.bias,
                                                activation=self.activation))
            for i in range(self.num_layers):
                setattr(self, f'res{i+1}', residual_layer(residual,
                                                          ch_list[i],
                                                          ch_list[i+1],
                                                          self.bias))

        elif self.mode // 100 == 1:
            # bert style
            self.num_layers = 1
            self.cnn1 = Conv(self.in_channels,
                             self.out_channels,
                             bias=self.bias,
                             normalization=lambda: self.normalization(
                                 self.out_channels),
                             dropout=self.dropout)
            self.res1 = null_fn

    def forward(self, x: Tensor) -> Tensor:
        # x: n,c,v,t
        x = self.norm(x)
        for i in range(self.num_layers):
            x = getattr(self, f'cnn{i+1}')(x) + getattr(self, f'res{i+1}')(x)
        return x


class FeatureExtractor(PyTorchModule):
    def __init__(self,
                 in_pos: int,
                 in_vel: int,
                 in_pos_emb_kwargs: dict,
                 in_vel_emb_kwargs: dict,
                 ):
        super(FeatureExtractor, self).__init__()
        self.in_pos = in_pos
        self.in_vel = in_vel
        if self.in_pos > 0:
            self.pos_embed = Embedding(**in_pos_emb_kwargs)
        if self.in_vel > 0:
            self.vel_embed = Embedding(**in_vel_emb_kwargs)
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


class SemanticEmbedding(PyTorchModule):
    def __init__(self,
                 num_point: int,
                 num_segment: int,
                 sem_spa: int,
                 sem_tem: int,
                 sem_spa_emb_kwargs: dict,
                 sem_tem_emb_kwargs: dict,
                 ):
        super(SemanticEmbedding, self).__init__()
        self.num_point = num_point
        self.num_segment = num_segment
        self.sem_spa = sem_spa
        self.sem_tem = sem_tem
        # Joint Embedding
        if self.sem_spa > 0:
            self.spa_onehot = OneHotTensor(
                sem_spa_emb_kwargs['in_channels'], self.num_segment, mode=0)
            self.spa_embedding = Embedding(**sem_spa_emb_kwargs)
        # Frame Embedding
        if self.sem_tem > 0:
            self.tem_onehot = OneHotTensor(
                sem_tem_emb_kwargs['in_channels'], self.num_point, mode=1)
            self.tem_embedding = Embedding(**sem_tem_emb_kwargs)

    def forward(self, x: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        spa_emb, tem_emb = None, None
        if self.sem_spa > 0:
            spa_emb = self.spa_embedding(self.spa_onehot(x.shape[0]))
        if self.sem_tem > 0:
            tem_emb = self.tem_embedding(self.tem_onehot(x.shape[0]))
        return spa_emb, tem_emb


class GCNSpatialG(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: int = 0,
                 activation: T1 = nn.Softmax,
                 g_proj_shared: bool = False,
                 ):
        super(GCNSpatialG, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=bias,
                                          activation=activation)
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
                 prenorm: bool = False,
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
        # in original SGN bias for w1 is false
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
                 g_proj_dim: T3 = 256,
                 g_kernel: int = 1,
                 g_proj_shared: bool = False,
                 g_activation: T1 = nn.Softmax,
                 return_g: bool = True,
                 return_gcn_list: bool = False,
                 ):
        super(GCNSpatialBlock, self).__init__()
        self.return_g = return_g
        self.return_gcn_list = return_gcn_list
        self.num_blocks = len(gcn_dims) - 1
        self.g_shared = isinstance(g_proj_dim, int)
        if self.g_shared:
            self.gcn_g1 = GCNSpatialG(gcn_dims[0],
                                      g_proj_dim,
                                      bias=bias,
                                      kernel_size=g_kernel,
                                      padding=g_kernel//2,
                                      activation=g_activation,
                                      g_proj_shared=g_proj_shared)
        else:
            for i in range(self.num_blocks):
                setattr(self, f'gcn_g{i+1}',
                        GCNSpatialG(gcn_dims[i],
                                    g_proj_dim[i],
                                    bias=bias,
                                    kernel_size=g_kernel,
                                    padding=g_kernel//2,
                                    activation=g_activation,
                                    g_proj_shared=g_proj_shared))

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
                                   prenorm=gcn_prenorm))

        if gcn_prenorm:
            for i in range(self.num_blocks):
                setattr(self, f'gcn_prenorm{i+1}', normalization(gcn_dims[i]))

        if isinstance(gcn_residual, list):
            assert len(gcn_residual) == self.num_blocks
            for i, r in enumerate(gcn_residual):
                setattr(self, f'gcn_res{i+1}',
                        residual_layer(r, gcn_dims[i], gcn_dims[i+1], bias))
            self.res = null_fn

        elif isinstance(gcn_residual, int):
            self.res = residual_layer(
                gcn_residual, gcn_dims[0], gcn_dims[-1], bias)

        else:
            raise ValueError("Unknown residual modes...")

    def forward(self, x: Tensor) -> Tensor:
        x0 = x
        g = []
        gcn_list = []
        for i in range(self.num_blocks):
            x1 = x

            if hasattr(self, f'gcn_prenorm{i+1}'):
                x1 = getattr(self, f'gcn_prenorm{i+1}')(x1)

            if (self.g_shared and len(g) == 0) or not self.g_shared:
                g1 = getattr(self, f'gcn_g{i+1}')(x1)
                g.append(g1)

            r = getattr(self, f'gcn_res{i+1}')(x)
            if isinstance(r, Tensor):
                if hasattr(self, f'gcn_maxpool{i+1}'):
                    r = getattr(self, f'gcn_maxpool{i+1}')(r)
            x = getattr(self, f'gcn{i+1}')(x1, g1) + r

            gcn_list.append(x)

        x += self.res(x0)

        if self.return_gcn_list and self.return_g:
            return x, g, gcn_list
        elif self.return_gcn_list:
            return x, gcn_list
        elif self.return_g:
            return x, g
        else:
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

        self.res = residual_layer(residual,
                                  channels[0], channels[-1], biases[0])

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

            setattr(self,
                    f'res{i+1}',
                    residual_layer(residuals[i],
                                   channels[i], channels[i+1], biases[i]))

    def forward(self, x: Tensor, x_n: Optional[Tensor] = None) -> Tensor:
        # x: n,c,v,t ; v=1 due to SMP
        x0 = x if x_n is None else x_n
        x = self.pool(x)
        for i in range(self.num_layers):
            x = getattr(self, f'cnn{i+1}')(x) + getattr(self, f'res{i+1}')(x)
        x += self.res(x0)
        return x


class MLPTemporalBranch(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,  # t_kernel
                 bias: int = 0,
                 dropout: Optional[Type[PyTorchModule]] = None,
                 activation: Optional[Type[PyTorchModule]] = None,
                 normalization: Optional[Type[PyTorchModule]] = None,
                 prenorm: bool = False,
                 t_mode: int = 0,
                 aspp_rates: Optional[List[int]] = None,
                 maxpool_kwargs: Optional[dict] = None,
                 ):
        super(MLPTemporalBranch, self).__init__(in_channels,
                                                out_channels,
                                                kernel_size=kernel_size,
                                                bias=bias,
                                                dropout=dropout,
                                                activation=activation,
                                                normalization=normalization,
                                                prenorm=prenorm)

        # aspp
        if aspp_rates is None or len(aspp_rates) == 0:
            self.aspp = nn.Identity()
        else:
            self.aspp = ASPP(self.in_channels,
                             self.in_channels,
                             bias=self.bias,
                             dilation=aspp_rates,
                             dropout=self.dropout,
                             activation=self.activation,
                             normalization=self.normalization)

        self.t_mode = t_mode
        assert t_mode in T_MODES

        # skip
        if t_mode == 0:
            self.cnn = nn.Identity()
        # original sgn
        elif t_mode == 1:
            idx = 2
            self.cnn = MLPTemporal(
                channels=[self.in_channels, self.in_channels,
                          self.out_channels],
                kernel_sizes=[self.kernel_size, 1],
                paddings=[self.kernel_size//2, 0],
                biases=[self.bias for _ in range(idx)],
                residuals=[0 for _ in range(idx)],
                dropouts=[self.dropout, None],
                activations=[self.activation for _ in range(idx)],
                normalizations=[self.normalization for _ in range(idx)],
                maxpool_kwargs=maxpool_kwargs,
                prenorm=self.prenorm
            )
        # original sgn with residual
        elif t_mode == 2:
            idx = 2
            self.cnn = MLPTemporal(
                channels=[self.in_channels, self.in_channels,
                          self.out_channels],
                kernel_sizes=[self.kernel_size, 1],
                paddings=[self.kernel_size//2, 0],
                biases=[self.bias for _ in range(idx)],
                residuals=[1 for _ in range(idx)],
                dropouts=[self.dropout, None],
                activations=[self.activation for _ in range(idx)],
                normalizations=[self.normalization for _ in range(idx)],
                maxpool_kwargs=maxpool_kwargs,
                prenorm=self.prenorm
            )
        else:
            raise ValueError('Unknown t_mode')

    def forward(self, x: Tensor) -> Tensor:
        # x: n,c,v,t ; v=1 due to SMP
        x = self.aspp(x)
        x = self.cnn(x)
        return x


if __name__ == '__main__':

    batch_size = 64

    inputs = torch.ones(batch_size, 20, 75)
    # subjects = torch.ones(batch_size, 40, 1)

    model = SGN(
        num_class=60,
        num_point=25,
        num_segment=20,
        in_channels=3,
        bias=1,
        dropout=0.0,  # classifier
        dropout2d=0.0,  # the rest
        c_multiplier=1,
        norm_type='bn-pre',
        act_type='relu',
        xem_projection=0,
        input_position=1,
        input_velocity=1,
        semantic_joint=1,
        semantic_frame=1,
        semantic_joint_fusion=0,
        semantic_frame_fusion=1,
        semantic_frame_location=0,
        sgcn_dims=None,  # [c2, c3, c3],
        sgcn_kernel=1,  # residual connection in GCN
        sgcn_padding=0,  # residual connection in GCN
        sgcn_dropout=0.0,  # residual connection in GCN
        # int for global res, list for individual gcn
        sgcn_residual=[0, 0, 0],
        sgcn_prenorm=False,
        sgcn_g_kernel=1,
        sgcn_g_proj_dim=None,  # c3
        sgcn_g_proj_shared=False,
        gcn_fpn=4,
        spatial_maxpool=1,
        temporal_maxpool=1,
        aspp_rates=None,
        t_mode=1,
        t_maxpool_kwargs=None,
        multi_t=[[], [], [3, 5, 7]],
        multi_t_shared=0,
    )
    model(inputs)
    print(model)

    try:
        flops = FlopCountAnalysis(model, inputs)
        print(flops.total())
        # print(flops.by_module_and_operator())
        # print(flop_count_table(flops))
    except NameError:
        print("Warning: fvcore is not found")