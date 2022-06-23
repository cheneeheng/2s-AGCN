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

import math
from typing import Tuple, Optional, Union, Type, List

# from model.module import *
# from model.module.bifpn import BiFPN
from model.resource.common_ntu import *
from model.module import Module
from model.module import BiFPN
from model.module import Conv
from model.module import ASPP
from model.module import null_fn
from model.module import init_zeros
from model.module import pad_zeros
from model.module import get_activation_fn
from model.module import get_normalization_fn
from model.module import tensor_list_mean
from model.module import tensor_list_sum
from model.module import Transformer
from utils.utils import *

T1 = Type[PyTorchModule]
T2 = List[Optional[Type[PyTorchModule]]]
T3 = Union[List[int], int]

EMB_MODES = [1, 2, 3, 4, 5, 6, 7, 8]

# GCN-FPN ---
# -1 no fpn
# 0 no fpn but has parallel projections
# 1 proj and sum
# 2 proj to lower and sum
# 3 proj
# 4 proj and concat
# 5 proj to lower and concat
# 6 proj to 64 and sum
# 7 proj with 1x3 and sum, similar to 1
# 8 bifpn 1 layer 64 dim
# 9 proj with 1x3 and sum, similar to 1, but witk K kernels and summed
GCN_FPN_MODES = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# how the gcn fpns are merged for fc.
# 0: no merge + no fpn
# 1: avraged and sent to 1 fc only.
# 2: no merge and used by multiple fc and averaged.
GCN_FPN_MERGE_MODES = [0, 1, 2]

# Temporal branch ---
# 0 skip -> no temporal branch
# 1 original sgn -> 1x3conv + 1x1conv
# 2 original sgn with residual -> 1x3conv + res + 1x1conv + res
# 3 trasnformers
T_MODES = [0, 1, 2, 3]

# Maxpooling ---
# 0 no pool
# 1 pool
# 2 pool with indices
POOLING_MODES = [0, 1, 2]


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


class SGN(PyTorchModule):

    # CONSTANTS
    ffn_mode = [0, 1, 2, 3, 101, 102, 103, 104, 201, 202]
    emb_modes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    c1, c2, c3, c4 = c1, c2, c3, c4
    g_activation_fn = nn.Identity  # nn.Softmax

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
                 sgcn_kernel: int = 1,  # res connection in GCN, 0 = no res
                 sgcn_padding: int = 0,  # res connection in GCN
                 sgcn_dropout: float = 0.0,  # res connection in GCN
                 # int for global res, list for individual gcn
                 sgcn_residual: T3 = [0, 0, 0],
                 sgcn_prenorm: bool = False,
                 sgcn_ffn: Optional[float] = None,
                 sgcn_v_kernel: int = 0,
                 sgcn_g_kernel: int = 1,
                 sgcn_g_proj_dim: Optional[T3] = None,  # c3
                 sgcn_g_proj_shared: bool = False,
                 sgcn_g_weighted: int = 0,

                 gcn_fpn: int = -1,
                 gcn_fpn_kernel: Union[int, list] = -1,
                 gcn_fpn_output_merge: int = 1,

                 bifpn_dim: int = 0,  # 64
                 bifpn_layers: int = 0,  # 1

                 spatial_maxpool: int = 1,
                 temporal_maxpool: int = 1,

                 # ORDER IS IMPORTANT
                 multi_t: List[List[int]] = [[], [], [3]],
                 multi_t_shared: int = 0,

                 t_mode: int = 1,
                 t_maxpool_kwargs: Optional[dict] = None,
                 t_mha_kwargs: Optional[dict] = None,
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

        (self.normalization_fn_1d,
         self.normalization_fn) = get_normalization_fn(norm_type)
        self.prenorm = True if 'pre' in norm_type else False

        self.activation_fn = get_activation_fn(act_type)

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

        # Spatial GCN ----------------------------------------------------------

        # Input dim to the GCN
        if self.semantic_joint == 0:
            self.gcn_in_ch = self.c1
        else:
            if self.semantic_joint_fusion == 0:
                self.gcn_in_ch = self.c1 * 2
            elif self.semantic_joint_fusion == 1:
                self.gcn_in_ch = self.c1

        # projection layer pre GCN
        if self.xem_projection > 0:
            self.x_emb_projection = Embedding(
                in_channels=self.gcn_in_ch,
                out_channels=self.gcn_in_ch,
                bias=self.bias,
                # dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
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
            gcn_v_kernel=sgcn_v_kernel,
            gcn_ffn=sgcn_ffn,
            g_kernel=sgcn_g_kernel,
            g_proj_dim=sgcn_g_proj_dim,
            g_proj_shared=sgcn_g_proj_shared,
            g_activation=self.g_activation_fn,
            g_weighted=sgcn_g_weighted,
            # return_g=True,
            # return_gcn_list=True,
        )

        # GCN FPN --------------------------------------------------------------
        self.gcn_fpn = gcn_fpn
        assert self.gcn_fpn in GCN_FPN_MODES

        self.gcn_fpn_output_merge = gcn_fpn_output_merge
        assert self.gcn_fpn_output_merge in GCN_FPN_MERGE_MODES

        if bifpn_dim > 0:
            assert self.gcn_fpn == 8

        self.gcn_fpn_kernel = gcn_fpn_kernel
        if isinstance(self.gcn_fpn_kernel, int):
            if self.gcn_fpn_kernel < 1:
                self.gcn_fpn_kernel = 1
            if self.gcn_fpn == 7:  # backward compat
                self.gcn_fpn_kernel = 3

        if self.gcn_fpn < 0:
            pass

        elif self.gcn_fpn == 0:
            if len(set(sgcn_dims)) == 1:
                pass
            else:
                assert self.semantic_frame_location == 1

        elif self.gcn_fpn == 8:
            self.bifpn = BiFPN(sgcn_dims, bifpn_dim, num_layers=bifpn_layers)

        elif self.gcn_fpn == 9:
            assert isinstance(self.gcn_fpn_kernel, list)
            for i in range(len(sgcn_dims)):
                for k in self.gcn_fpn_kernel:
                    setattr(self,
                            f'fpn_proj{i+1}_k{k}',
                            Conv(sgcn_dims[i],
                                 sgcn_dims[-1],
                                 kernel_size=k,
                                 padding=k//2,
                                 bias=self.bias,
                                 activation=self.activation_fn,
                                 normalization=lambda: self.normalization_fn(
                                sgcn_dims[-1]),
                                #  dropout=self.dropout_fn,
                            ))

        else:
            assert isinstance(self.gcn_fpn_kernel, int)
            for i in range(len(sgcn_dims)):
                if self.gcn_fpn in [1, 3, 4, 7]:
                    out_channels = sgcn_dims[-1]
                elif self.gcn_fpn == 2:
                    out_channels = sgcn_dims[0]
                elif self.gcn_fpn == 5:
                    out_channels = sgcn_dims[-1]//4
                elif self.gcn_fpn == 6:
                    out_channels = 64
                else:
                    raise ValueError
                setattr(self,
                        f'fpn_proj{i+1}',
                        Conv(sgcn_dims[i],
                             out_channels,
                             kernel_size=self.gcn_fpn_kernel,
                             padding=self.gcn_fpn_kernel//2,
                             bias=self.bias,
                             activation=self.activation_fn,
                             normalization=lambda: self.normalization_fn(
                                 out_channels),
                             #  dropout=self.dropout_fn,
                             ))

        # Joint and frame embeddings -------------------------------------------
        # post gcn
        if self.semantic_frame_location == 0:
            if self.gcn_fpn == 2:
                out_channels = self.gcn_in_ch
            elif self.gcn_fpn == 5:
                out_channels = sgcn_dims[-1]//4
            elif self.gcn_fpn == 6:
                out_channels = 64
            elif self.gcn_fpn == 8:
                out_channels = bifpn_dim
            else:
                out_channels = sgcn_dims[-1]
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

        # Frame level module ---------------------------------------------------
        self.t_mode = t_mode
        self.t_maxpool_kwargs = t_maxpool_kwargs
        self.t_mha_kwargs = t_mha_kwargs
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
                elif self.gcn_fpn in [1, 3, 7, 9]:
                    in_ch = sgcn_dims[-1]
                elif self.gcn_fpn == 2:
                    in_ch = sgcn_dims[0]
                elif self.gcn_fpn == 4:
                    in_ch = sgcn_dims[-1]*3
                elif self.gcn_fpn == 5:
                    in_ch = sgcn_dims[-1]//4 * 3
                elif self.gcn_fpn == 6:
                    in_ch = 64
                elif self.gcn_fpn == 8:
                    in_ch = bifpn_dim
                else:
                    in_ch = sgcn_dims[-1]

                if self.t_mode == 3:
                    name = f'tem_mha_{i+1}_{j+1}'
                else:
                    name = f'tem_mlp_{i+1}_{j+1}_k{t_kernel}'

                if self.multi_t_shared == 2:
                    cont = False
                    for k in range(i+1):
                        if self.t_mode == 3:
                            name_k = f'tem_mha_{k+1}_{j+1}'
                        else:
                            name_k = f'tem_mlp_{k+1}_{j+1}_k{t_kernel}'
                        if getattr(self, name_k, None) is not None:
                            cont = True
                    if cont:
                        continue

                setattr(self,
                        name,
                        TemporalBranch(
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
                            mha_kwargs=self.t_mha_kwargs,
                        ))

                # if self.multi_t_shared == 1:
                #     break

        # Maxpool --------------------------------------------------------------
        # 0 no pool, 1 pool, 2 pool with indices
        self.spatial_maxpool = spatial_maxpool
        self.temporal_maxpool = temporal_maxpool
        assert self.spatial_maxpool in POOLING_MODES
        assert self.temporal_maxpool in POOLING_MODES

        if self.spatial_maxpool == 0:
            self.smp = nn.Identity()
        elif self.spatial_maxpool == 1:
            self.smp = nn.AdaptiveMaxPool2d((1, self.num_segment))
        elif self.spatial_maxpool == 2:
            raise ValueError("spatial_maxpool=2 not implemented")
        else:
            raise ValueError("Unknown spatial_maxpool")

        if self.temporal_maxpool == 0:
            self.tmp = nn.Identity()
        elif self.temporal_maxpool == 1:
            self.tmp = nn.AdaptiveMaxPool2d((1, 1))
        elif self.temporal_maxpool == 2:
            self.tmp = nn.AdaptiveMaxPool2d((1, 1), return_indices=True)
        else:
            raise ValueError("Unknown temporal_maxpool")

        # Classifier ---------------------------------------------------------
        fc_in_ch = self.c4  # default
        if self.t_mode == 0:
            fc_in_ch = self.c3
        if self.t_mode == 3:
            fc_in_ch = self.t_mha_kwargs.get('dim_feedforward_output', None)
            if fc_in_ch is None:
                fc_in_ch = self.t_mha_kwargs.get('d_model', None)
            if fc_in_ch is None:
                raise ValueError("dim_feedforward_output/d_model missing...")
            if isinstance(fc_in_ch, list):
                fc_in_ch = fc_in_ch[-1]
        if self.spatial_maxpool == 0 and self.temporal_maxpool == 0:
            fc_in_ch = fc_in_ch * self.num_segment * self.num_point
        if self.temporal_maxpool == 0:
            fc_in_ch = fc_in_ch * self.num_segment

        if self.gcn_fpn_output_merge == 2:
            i = 0
            for _ in self.multi_t:
                for _ in t_kernels:
                    i += 1
                    setattr(self, f'fc{i}', nn.Linear(fc_in_ch, num_class))
        else:
            self.fc = nn.Linear(fc_in_ch, num_class)

        if self.temporal_maxpool == 2:
            self.tmp_ind_projection = Embedding(
                in_channels=fc_in_ch,
                out_channels=fc_in_ch,
                bias=self.bias,
                activation=self.activation_fn,
                mode=1
            )

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
                init_zeros(param)

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
        elif self.gcn_fpn == 9:
            assert hasattr(self, 'sgcn')
            x_list = [
                tensor_list_sum(
                    [getattr(self, f'fpn_proj{i+1}_k{k}')(x_spa_list[i])
                     for k in self.gcn_fpn_kernel])
                for i in range(len(x_spa_list))
            ]
            x_list = [tensor_list_sum(x_list[i:]) for i in range(len(x_list))]
        elif self.gcn_fpn in [1, 2, 6, 7]:
            assert hasattr(self, 'sgcn')
            x_list = [getattr(self, f'fpn_proj{i+1}')(x_spa_list[i])
                      for i in range(len(x_spa_list))]
            x_list = [tensor_list_sum(x_list[i:]) for i in range(len(x_list))]
        elif self.gcn_fpn in [3, 4, 5]:
            assert hasattr(self, 'sgcn')
            x_list = [getattr(self, f'fpn_proj{i+1}')(x_spa_list[i])
                      for i in range(len(x_spa_list))]
        elif self.gcn_fpn == 8:
            assert hasattr(self, 'sgcn')
            assert hasattr(self, 'bifpn')
            x_list = self.bifpn(x_spa_list)
        else:
            x_list = [None for _ in range(len(x_spa_list)-1)] + \
                     [x_spa_list[-1]]

        # Frame-level Module ---------------------------------------------------
        # temporal fusion post gcn
        if self.semantic_frame > 0 and self.semantic_frame_location == 0:
            x_list = [i + tem_emb if i is not None else None for i in x_list]

        # spatial pooling
        x_list = [self.smp(i) if i is not None else None for i in x_list]

        if self.gcn_fpn in [4, 5]:
            x_list = [None for _ in range(len(x_spa_list)-1)] + \
                     [torch.cat(x_list, dim=1)]

        # temporal MLP
        _x_list = []
        for i, t_kernels in enumerate(self.multi_t):
            for j, t_kernel in enumerate(t_kernels):

                if x_list[i] is None:
                    continue

                if self.t_mode == 3:
                    name = f'tem_mha_{i+1}_{j+1}'
                else:
                    name = f'tem_mlp_{i+1}_{j+1}_k{t_kernel}'

                # if self.multi_t_shared == 1:
                #     name = f'tem_mlp_{i+1}_{1}_k{t_kernel}'
                if self.multi_t_shared == 2:
                    for k in range(i):
                        if self.t_mode == 3:
                            name_k = f'tem_mha_{k+1}_{j+1}'
                        else:
                            name_k = f'tem_mlp_{k+1}_{j+1}_k{t_kernel}'
                        if getattr(self, name_k, None) is not None:
                            name = name_k
                            break

                _x_list.append(getattr(self, name)(x_list[i]))

        if self.gcn_fpn_output_merge in [0, 1]:
            x = tensor_list_mean(_x_list)
        elif self.gcn_fpn_output_merge == 2:
            x = _x_list
        else:
            raise ValueError("Unknown 'gcn_fpn_output_merge' arg value...")

        # temporal pooling
        if isinstance(x, list):
            y = [self.tmp(i) for i in x]
        else:
            y = self.tmp(x)

        if self.temporal_maxpool == 2:
            if isinstance(y, list):
                y_ind = [i[1] for i in y]
                y = [i[0] for i in y]
                y = [i + self.tmp_ind_projection(j.float())
                     for i, j in zip(y, y_ind)]
            else:
                y, y_ind = y
                y = y + self.tmp_ind_projection(y_ind.float())

        # Classification -------------------------------------------------------
        if isinstance(x, list):
            if self.gcn_fpn_output_merge == 2:
                _y_list = []
                for i, y_i in enumerate(y):
                    y_i = torch.flatten(y_i, 1)
                    y_i = self.fc_dropout(y_i)
                    y_i = getattr(self, f'fc{i+1}')(y_i)
                    _y_list.append(y_i)
                y = tensor_list_mean(_y_list)
        else:
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
        return x


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
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, g: Optional[Tensor] = None) -> Tensor:
        g1 = self.g1(x).permute(0, 3, 2, 1).contiguous()  # n,t,v,c
        g2 = self.g2(x).permute(0, 3, 1, 2).contiguous()  # n,t,c,v
        g3 = g1.matmul(g2)  # n,t,v,v
        g4 = self.act(g3)
        if g is not None:
            g4 = (g * self.alpha + g4) / (self.alpha + 1)
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
                 v_kernel_size: int = 0,
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
        if v_kernel_size > 0:
            self.w0 = Conv(self.in_channels,
                           self.in_channels,
                           kernel_size=v_kernel_size,
                           padding=v_kernel_size//2,
                           bias=self.bias)
        else:
            self.w0 = nn.Identity()
        # in original SGN bias for w1 is false
        self.w1 = Conv(self.in_channels, self.out_channels, bias=self.bias)
        if self.kernel_size > 0:
            self.w2 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                           kernel_size=self.kernel_size, padding=self.padding)
        else:
            self.w2 = null_fn
        if self.prenorm:
            self.norm = nn.Identity()
        else:
            self.norm = self.normalization(self.out_channels)
        self.act = self.activation()
        self.drop = nn.Identity() if self.dropout is None else self.dropout()

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        x1 = self.w0(x)
        x1 = x1.permute(0, 3, 2, 1).contiguous()  # n,t,v,c
        x1 = g.matmul(x1)
        x1 = x1.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
        x1 = self.w1(x1) + self.w2(x)  # z + residual
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
                 gcn_v_kernel: int = 0,
                 gcn_ffn: Optional[float] = None,
                 g_proj_dim: T3 = 256,
                 g_kernel: int = 1,
                 g_proj_shared: bool = False,
                 g_activation: T1 = nn.Softmax,
                 g_weighted: int = 0,
                 #  return_g: bool = True,
                 #  return_gcn_list: bool = False,
                 ):
        super(GCNSpatialBlock, self).__init__()
        # self.return_g = return_g
        # self.return_gcn_list = return_gcn_list

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
                                   v_kernel_size=gcn_v_kernel))

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
                        residual_layer(r, gcn_dims[i], gcn_dims[i+1], bias))
            self.res = null_fn

        elif isinstance(gcn_residual, int):
            self.res = residual_layer(
                gcn_residual, gcn_dims[0], gcn_dims[-1], bias)

        else:
            raise ValueError("Unknown residual modes...")

    def forward(self, x: Tensor) -> Tuple[Tensor, list, list]:
        x0 = x
        g = []
        gcn_list = []
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
            x = getattr(self, f'gcn{i+1}')(x1, g[-1]) + r

            if hasattr(self, f'gcn_ffn{i+1}'):
                x = getattr(self, f'gcn_ffn{i+1}')(x)

            gcn_list.append(x)

        x += self.res(x0)

        return x, g, gcn_list
        # if self.return_gcn_list and self.return_g:
        #     return x, g, gcn_list
        # elif self.return_gcn_list:
        #     return x, gcn_list
        # elif self.return_g:
        #     return x, g
        # else:
        #     return x


class MHATemporal(PyTorchModule):
    def __init__(self, kwargs: dict):
        super(MHATemporal, self).__init__()
        if 'norm' in kwargs:
            self.transformer = Transformer(
                dim=kwargs['d_model'],
                depth=kwargs['num_layers'],
                heads=kwargs['nhead'],
                dim_head=kwargs['d_head'],
                dropout=kwargs['dropout'],
                mlp_dim=kwargs['dim_feedforward'],
                mlp_out_dim=kwargs['dim_feedforward_output'],
                activation=kwargs['activation'],
                norm=kwargs['norm'],
                global_norm=kwargs['global_norm']
            )
        else:
            self.num_layers = kwargs['num_layers']
            for i in range(self.num_layers):
                setattr(self,
                        f'layer{i+1}',
                        nn.TransformerEncoderLayer(
                            d_model=kwargs['d_model'],
                            nhead=kwargs['nhead'],
                            dim_feedforward=kwargs['dim_feedforward'],
                            dropout=kwargs['dropout'],
                            activation=kwargs['activation'],
                            layer_norm_eps=1e-5,
                            batch_first=True,
                        ))

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, 'transformer'):
            x, _ = self.transformer(x)
        else:
            for i in range(self.num_layers):
                x = getattr(self, f'layer{i+1}')(x)
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


class TemporalBranch(Module):
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
                 mha_kwargs: Optional[dict] = None,
                 ):
        super(TemporalBranch, self).__init__(in_channels,
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

        # Temporal branch ------------------------------------------------------
        self.t_mode = t_mode
        assert t_mode in T_MODES

        if t_mode == 0:
            self.cnn = nn.Identity()
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
        elif t_mode == 3:
            self.cnn = MHATemporal(mha_kwargs)
        else:
            raise ValueError('Unknown t_mode')

    def forward(self, x: Tensor) -> Tensor:
        N, _, _, T = x.shape
        x = self.aspp(x)
        if self.t_mode == 3:
            x = x.permute(0, 3, 2, 1).contiguous()  # n,t,v,c
            x = x.reshape(N, T, -1)
        x = self.cnn(x)
        if self.t_mode == 3:
            x = x.reshape(N, T, 1, -1)
            x = x.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
        return x


if __name__ == '__main__':

    batch_size = 1

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
        c_multiplier=[1.0, 1.0, 1.0, 1.0],
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
        sgcn_dims=[128, 256, 256],  # [c2, c3, c3],
        sgcn_kernel=1,  # residual connection in GCN
        sgcn_padding=0,  # residual connection in GCN
        sgcn_dropout=0.0,  # residual connection in GCN
        # int for global res, list for individual gcn
        sgcn_residual=[0, 0, 0],
        sgcn_prenorm=False,
        # sgcn_ffn=0,
        sgcn_v_kernel=0,
        sgcn_g_kernel=1,
        sgcn_g_proj_dim=256,  # c3
        sgcn_g_proj_shared=False,
        # sgcn_g_weighted=1,
        gcn_fpn=1,
        gcn_fpn_kernel=3,
        # gcn_fpn_output_merge=1,
        # bifpn_dim=256,
        # bifpn_layers=1,
        spatial_maxpool=1,
        temporal_maxpool=1,
        aspp_rates=None,
        t_mode=1,
        # t_maxpool_kwargs=None,
        t_mha_kwargs={
            'd_model': [256, 512],
            'nhead': [1, 1],
            'd_head': [256, 512],
            'dim_feedforward': [256, 512],
            'dim_feedforward_output': [512, 1024],
            'dropout': 0.1,
            'activation': "relu",
            'num_layers': 2,
            'norm': 'ln',
            'global_norm': False
        },
        multi_t=[[3, 5, 7], [3, 5, 7], [3, 5, 7]],
        multi_t_shared=2,
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
