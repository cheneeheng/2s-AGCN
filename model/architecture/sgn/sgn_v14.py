# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Taken from :
# https://github.com/microsoft/SGN/blob/master/model.py

# Code refractored

# Continue from on sgn_v13
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
from collections import OrderedDict
from typing import Tuple, Optional, Union, Type, List

from model.resource.common_ntu import *
from model.layers import BiFPN
from model.layers import Conv
from model.layers import fuse_features
from model.layers import init_zeros
from model.layers import pad_zeros
from model.layers import get_activation_fn
from model.layers import get_normalization_fn
from model.layers import tensor_list_mean
from model.layers import tensor_list_sum
from utils.utils import to_int

from model.architecture.sgn.blocks import Embedding
from model.architecture.sgn.blocks import SemanticEmbedding
from model.architecture.sgn.blocks import TemporalBranch
from model.architecture.sgn.blocks import GCNSpatialBlock
from model.architecture.sgn.blocks import GCNSpatialBlock2


T1 = Type[PyTorchModule]
T2 = List[Optional[Type[PyTorchModule]]]
T3 = Union[List[int], int]


EMB_MODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]

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
# 10 second sgcn in fpn style
GCN_FPN_MODES = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# how the gcn fpns are merged for fc.
# 0: no merge + no fpn (not needed, use 1)
# 1: avraged and sent to 1 fc only.
# 2: no merge and used by multiple fc and averaged.
GCN_FPN_MERGE_MODES = [0, 1, 2]

# Maxpooling ---
# 0 no pool
# 1 pool
# 2 pool with indices
# 3 conv with large kernel
# 4 1x1 + conv with large kernel
POOLING_MODES = [0, 1, 2, 3, 4]

# Temporal branch ---
# 0 skip -> no temporal branch
# 1 original sgn -> 1x3conv + 1x1conv
# 2 original sgn with residual -> 1x3conv + res + 1x1conv + res
# 3 trasnformers
# 4 decompose + original sgn -> 1x3conv + 1x1conv
# 5 avg pool + cnn
T_MODES = [0, 1, 2, 3, 4, 5]


class SGN(PyTorchModule):

    # CONSTANTS
    ffn_mode = [0, 1, 2, 3, 101, 102, 103, 104, 201, 202]
    emb_modes = EMB_MODES  # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    c1, c2, c3, c4 = c1, c2, c3, c4
    # g_activation_fn = nn.Identity
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
                 semantic_class: int = 0,
                 semantic_joint_smp: int = 0,

                 semantic_joint_fusion: int = 0,
                 semantic_frame_fusion: int = 1,
                 semantic_frame_location: int = 0,
                 semantic_class_fusion: int = 0,

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
                 sgcn_g_res_alpha: float = 1.0,
                 # 1: matmul GT @ G
                 # 2: pointwise mul post GT-softmax GT * G
                 # 3: pointwise mul post GT-softmax/sigmoid (2 layers) GT * G
                 # 4: MLP for GT and used multiplied t input of GCN GT, G
                 # 5: MLP for GT and used together for final prediction GT, G
                 # 6: MLP for GT and used together for final prediction GT, G
                 #    The GT is stacked with FPN list for multikernel pred.
                 sgcn_gt_mode: int = 1,
                 # 1: softmax
                 # 2: sigmoid
                 sgcn_gt_act: int = 1,
                 sgcn_gt_g3_idx: int = 2,
                 # 0: mat mul
                 # 1: w1,w2 linear proj
                 # 2: squeeze excite
                 # 3: only w2
                 sgcn_attn_mode: int = 0,
                 sgcn_gt_out_channels2: int = 512,

                 sgcn2_dims: Optional[list] = None,  # [c2, c3, c3],
                 sgcn2_kernel: int = 1,  # res connection in GCN, 0 = no res
                 sgcn2_padding: int = 0,  # res connection in GCN
                 sgcn2_dropout: float = 0.0,  # res connection in GCN
                 # int for global res, list for individual gcn
                 sgcn2_residual: T3 = [0, 0, 0],
                 sgcn2_prenorm: bool = False,
                 sgcn2_ffn: Optional[float] = None,
                 sgcn2_v_kernel: int = 0,
                 sgcn2_g_kernel: int = 1,
                 sgcn2_g_proj_dim: Optional[T3] = None,  # c3
                 sgcn2_g_proj_shared: bool = False,
                 sgcn2_g_weighted: int = 0,
                 sgcn2_g_res_alpha: float = 1.0,
                 sgcn2_gt_mode: int = 1,
                 sgcn2_gt_act: int = 1,
                 sgcn2_gt_g3_idx: int = 2,
                 sgcn2_attn_mode: int = 0,

                 gcn_fpn: int = -1,
                 gcn_fpn_kernel: Union[int, list] = -1,
                 gcn_fpn_output_merge: int = 1,
                 gcn_fpn_shared: int = 0,

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

                 decomp_kernel_size: int = 3,
                 pool_kernel_sizes: List[int] = [3, 5, 7, 9],
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
        self.semantic_class = semantic_class
        self.semantic_joint_smp = semantic_joint_smp
        self.xem_projection = xem_projection  # projection layer pre GCN
        assert self.input_position in self.emb_modes
        assert self.input_velocity in self.emb_modes
        assert self.semantic_joint in self.emb_modes
        assert self.semantic_frame in self.emb_modes
        assert self.semantic_class in self.emb_modes
        assert self.semantic_joint_smp in self.emb_modes
        assert self.xem_projection in self.emb_modes
        if self.input_position == 0 and self.semantic_joint > 0:
            raise ValueError("input_position is 0 but semantic_joint is not")

        # 0: concat, 1: sum
        self.semantic_joint_fusion = semantic_joint_fusion
        # 0: concat, 1: sum
        self.semantic_frame_fusion = semantic_frame_fusion  # UNUSED
        # 0 = add after GCN, 1 = add before GCN
        self.semantic_frame_location = semantic_frame_location
        assert self.semantic_frame_location in [0, 1]
        # 0: concat, 1: sum
        self.semantic_class_fusion = semantic_class_fusion  # UNUSED

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
                mode=self.input_position,
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
                mode=self.input_velocity,
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
        self.sgcn_gt_mode = sgcn_gt_mode
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
            gcn_attn_mode=sgcn_attn_mode,
            gcn_ffn=sgcn_ffn,
            g_kernel=sgcn_g_kernel,
            g_proj_dim=sgcn_g_proj_dim,
            g_proj_shared=sgcn_g_proj_shared,
            g_activation=self.g_activation_fn,
            g_weighted=sgcn_g_weighted,
            g_num_segment=self.num_segment,
            g_num_joint=self.num_point,
            g_res_alpha=sgcn_g_res_alpha,
            gt_mode=sgcn_gt_mode,
            gt_act=sgcn_gt_act,
            gt_g3_idx=sgcn_gt_g3_idx,
            gt_out_channels2=sgcn_gt_out_channels2
        )
        if sgcn2_dims is not None:
            self.sgcn2 = GCNSpatialBlock2(
                kernel_size=sgcn2_kernel,
                padding=sgcn2_padding,
                bias=self.bias,
                dropout=lambda: nn.Dropout2d(sgcn2_dropout),
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                gcn_dims=[sgcn_dims[-1]] + sgcn2_dims,
                gcn_residual=sgcn2_residual,
                gcn_prenorm=sgcn2_prenorm,
                gcn_v_kernel=sgcn2_v_kernel,
                gcn_attn_mode=sgcn2_attn_mode,
                gcn_ffn=sgcn2_ffn,
                g_kernel=sgcn2_g_kernel,
                g_proj_dim=sgcn2_g_proj_dim,
                g_proj_shared=sgcn2_g_proj_shared,
                g_activation=self.g_activation_fn,
                g_weighted=sgcn2_g_weighted,
                g_num_segment=self.num_segment,
                g_num_joint=self.num_point,
                g_res_alpha=sgcn2_g_res_alpha,
                gt_mode=sgcn2_gt_mode,
                gt_act=sgcn2_gt_act,
                gt_g3_idx=sgcn2_gt_g3_idx
            )

        # GCN FPN --------------------------------------------------------------
        self.gcn_fpn = gcn_fpn
        assert self.gcn_fpn in GCN_FPN_MODES

        self.gcn_fpn_output_merge = gcn_fpn_output_merge
        assert self.gcn_fpn_output_merge in GCN_FPN_MERGE_MODES

        # not for mode 8
        # if used for the rest, the gcn dim must all be 256
        self.gcn_fpn_shared = gcn_fpn_shared

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

        elif self.gcn_fpn == 10:
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
                    if self.gcn_fpn_shared == 1:
                        cont = False
                        for j in range(i+1):
                            name_k = f'fpn_proj{j+1}_k{k}'
                            if getattr(self, name_k, None) is not None:
                                cont = True
                        if cont:
                            continue
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
                if self.gcn_fpn_shared == 1:
                    cont = False
                    for j in range(i+1):
                        name_k = f'fpn_proj{j+1}'
                        if getattr(self, name_k, None) is not None:
                            cont = True
                    if cont:
                        continue
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
            sem_cls=self.semantic_class,
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
            ),
            sem_cls_emb_kwargs=dict(
                in_channels=1,
                out_channels=self.c4,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                num_point=self.num_point,
                mode=self.semantic_class
            )
        )

        if self.semantic_joint_smp > 0:
            self.semantic_joint_smp_embedding = SemanticEmbedding(
                num_point=self.num_point,
                num_segment=self.num_segment,
                sem_spa=self.semantic_joint_smp,
                sem_tem=0,
                sem_cls=0,
                sem_spa_emb_kwargs=dict(
                    in_channels=self.num_point,
                    out_channels=out_channels,
                    bias=self.bias,
                    dropout=self.dropout_fn,
                    activation=self.activation_fn,
                    normalization=self.normalization_fn,
                    num_point=self.num_point,
                    mode=self.semantic_joint_smp
                ),
                sem_tem_emb_kwargs=dict(),
                sem_cls_emb_kwargs=dict()
            )

        # Frame level module ---------------------------------------------------
        self.decomp_kernel_size = decomp_kernel_size
        self.pool_kernel_sizes = pool_kernel_sizes

        self.t_mode = t_mode
        assert t_mode in T_MODES
        self.t_maxpool_kwargs = t_maxpool_kwargs
        self.t_mha_kwargs = t_mha_kwargs
        self.aspp_rates = aspp_rates  # dilation rates
        self.multi_t = multi_t  # list of list : gcn layers -> kernel sizes
        if self.sgcn_gt_mode == 6:
            assert len(self.multi_t) == len(sgcn_dims) + 1
        else:
            assert len(self.multi_t) == len(sgcn_dims)
        # 0: no
        # 1: intra gcn share (no sense)
        # 2: inter gcn share (between layer share)
        self.multi_t_shared = multi_t_shared
        assert self.multi_t_shared in [0, 2]

        # loop through the gcn fpn
        if self.sgcn_gt_mode == 6:
            _dims = self.sgcn_dims + [self.num_point**2]
        else:
            _dims = self.sgcn_dims
        for i, (sgcn_dim, t_kernels) in enumerate(zip(_dims, self.multi_t)):

            # # all the kernel size for that gcn layer should be the same
            # if self.multi_t_shared == 1 and len(t_kernels) > 0:
            #     assert len(set(t_kernels)) == 1

            for j, t_kernel in enumerate(t_kernels):

                if self.sgcn_gt_mode == 6 and i == len(_dims)-1:
                    def_in_ch = _dims[-1]
                else:
                    def_in_ch = sgcn_dims[-1]

                if self.gcn_fpn == 0:
                    in_ch = sgcn_dim
                elif self.gcn_fpn in [1, 3, 7, 9]:
                    in_ch = def_in_ch
                elif self.gcn_fpn == 2:
                    in_ch = sgcn_dims[0]
                elif self.gcn_fpn == 4:
                    in_ch = def_in_ch*3
                elif self.gcn_fpn == 5:
                    in_ch = def_in_ch//4 * 3
                elif self.gcn_fpn == 6:
                    in_ch = 64
                elif self.gcn_fpn == 8:
                    in_ch = bifpn_dim
                elif self.gcn_fpn == 10:
                    in_ch = sgcn2_dims[i]
                else:
                    in_ch = def_in_ch

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
                            decomp_kernel_size=self.decomp_kernel_size,
                            pool_kernel_sizes=self.pool_kernel_sizes,
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
        elif self.spatial_maxpool == 3:
            self.smp = Conv(
                in_channels=self.c3*2 if self.semantic_joint_smp > 0 else self.c3,  # noqa
                out_channels=self.c3,
                kernel_size=self.num_point,
                bias=self.bias,
                activation=self.activation_fn,
                normalization=lambda: self.normalization_fn(
                    out_channels)
            )
        elif self.spatial_maxpool == 4:
            self.smp = nn.Sequential(
                OrderedDict([
                    (f'conv1', Conv(in_channels=self.c3*2 if self.semantic_joint_smp > 0 else self.c3,  # noqa
                                    out_channels=self.c3,
                                    kernel_size=1,
                                    bias=self.bias,
                                    activation=self.activation_fn,
                                    normalization=lambda: self.normalization_fn(out_channels))),  # noqa
                    (f'conv2', Conv(in_channels=self.c3,
                                    out_channels=self.c3,
                                    kernel_size=self.num_point,
                                    bias=self.bias,
                                    activation=self.activation_fn,
                                    normalization=lambda: self.normalization_fn(out_channels))),  # noqa
                ])
            )
            self.smp = Conv(
                in_channels=self.c3*2 if self.semantic_joint_smp > 0 else self.c3,  # noqa
                out_channels=self.c3,
                kernel_size=self.num_point,
                bias=self.bias,
                activation=self.activation_fn,
                normalization=lambda: self.normalization_fn(
                    out_channels)
            )
        else:
            raise ValueError("Unknown spatial_maxpool")

        # if self.semantic_joint_smp > 0:
        #     self.smp.return_indices = True

        if self.temporal_maxpool == 0:
            self.tmp = nn.Identity()
        elif self.temporal_maxpool == 1:
            self.tmp = nn.AdaptiveMaxPool2d((1, 1))
        elif self.temporal_maxpool == 2:
            self.tmp = nn.AdaptiveMaxPool2d((1, 1), return_indices=True)
        elif self.temporal_maxpool == 3:
            raise ValueError("temporal_maxpool=3 not implemented")
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
        bs, step, dim = x.shape
        x = x.view((bs, step, self.num_point, dim//self.num_point))  # n,t,v,c
        x = x.permute(0, 3, 2, 1).contiguous()  # n,c,v,t

        if x.shape[1] < self.in_channels:
            raise ValueError("tensor x has lower ch dim than self.in_channels")
        elif x.shape[-1] > self.in_channels:
            # print("tensor x has more ch dim than self.in_channels")
            # bs, step, dim = x.shape
            x = x[:, :self.in_channels, :, :]

        # Dynamic Representation -----------------------------------------------
        x = self.feature_extractor(x)
        assert x is not None

        # Joint and frame embeddings -------------------------------------------
        spa_emb, tem_emb, cls_emb = self.semantic_embedding(x)

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
        _, g_spa, x_spa_list, featuremap_spa_list = self.sgcn(x)

        # gcn fpn
        if self.gcn_fpn == 0:
            x_list = x_spa_list
        elif self.gcn_fpn == 9:
            assert hasattr(self, 'sgcn')
            x_list = [
                tensor_list_sum([
                    getattr(self, f'fpn_proj{i+1}_k{k}',
                            getattr(self, f'fpn_proj1_k{k}'))(x_spa_list[i])
                    for k in self.gcn_fpn_kernel
                ])
                for i in range(len(x_spa_list))
            ]
            x_list = [tensor_list_sum(x_list[i:]) for i in range(len(x_list))]
        elif self.gcn_fpn in [1, 2, 6, 7]:
            assert hasattr(self, 'sgcn')
            x_list = [
                getattr(self, f'fpn_proj{i+1}',
                        getattr(self, 'fpn_proj1'))(x_spa_list[i])
                for i in range(len(x_spa_list))
            ]
            x_list = [tensor_list_sum(x_list[i:]) for i in range(len(x_list))]
        elif self.gcn_fpn in [3, 4, 5]:
            assert hasattr(self, 'sgcn')
            x_list = [
                getattr(self, f'fpn_proj{i+1}',
                        getattr(self, 'fpn_proj1'))(x_spa_list[i])
                for i in range(len(x_spa_list))
            ]
        elif self.gcn_fpn == 8:
            assert hasattr(self, 'sgcn')
            assert hasattr(self, 'bifpn')
            x_list = self.bifpn(x_spa_list)
        elif self.gcn_fpn == 10:
            _x_list = [x] + x_spa_list[:-1]
            _x_list.reverse()
            _, g_spa2, x_spa_list2, featuremap_spa_list2 = self.sgcn2(
                x_spa_list[-1], _x_list, g_spa[-1])
            x_list = [None for _ in range(len(x_spa_list2)-1)] + \
                [x_spa_list2[-1]]
        else:
            x_list = [None for _ in range(len(x_spa_list)-1)] + \
                [x_spa_list[-1]]

        # Frame-level Module ---------------------------------------------------
        # temporal fusion post gcn
        if self.semantic_frame > 0 and self.semantic_frame_location == 0:
            x_list = [i + tem_emb if i is not None else None for i in x_list]

        # spatial pooling
        if hasattr(self, 'semantic_joint_smp_embedding'):
            smp_emb = self.semantic_joint_smp_embedding(x)[0]
            x_list = [torch.cat((i, smp_emb), axis=1)
                      if i is not None else None for i in x_list]

        if self.spatial_maxpool in [3, 4]:
            x_list = [i.permute(0, 1, 3, 2).contiguous()
                      if i is not None else None for i in x_list]
            x_list = [self.smp(i) if i is not None else None for i in x_list]
            x_list = [i.permute(0, 1, 3, 2).contiguous()
                      if i is not None else None for i in x_list]
        else:
            x_list = [self.smp(i) if i is not None else None for i in x_list]

        if self.gcn_fpn in [4, 5]:
            x_list = [None for _ in range(len(x_spa_list)-1)] + \
                [torch.cat(x_list, dim=1)]

        if self.sgcn_gt_mode == 6:
            x_list.append(g_spa[0][1])

        # temporal MLP
        _x_list = []
        attn_tem_list = []
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

                tem_out = getattr(self, name)(x_list[i])

                if isinstance(tem_out[0], list):
                    _x_list += tem_out[0]
                else:
                    _x_list.append(tem_out[0])

                attn_tem_list.append(tem_out[1])

        if self.sgcn_gt_mode == 5:
            _x_list.append(g_spa[0][1])

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

        if cls_emb is not None:
            if isinstance(y, list):
                y = [i + cls_emb for i in y]
            else:
                y += cls_emb

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
        if isinstance(y, list):
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

        if self.gcn_fpn == 10:
            return (
                y,
                {
                    'attn_tem_list': attn_tem_list,
                    'g_spa': g_spa,
                    'g_spa2': g_spa2,
                    'featuremap_spa_list': featuremap_spa_list,
                    'featuremap_spa_list2': featuremap_spa_list2,
                    'x_spa_list': x_spa_list,
                    'x_spa_list2': x_spa_list2,
                    'x_tem_list': _x_list,
                    'tem_emb': tem_emb,
                    'spa_emb': spa_emb,
                }
            )
        else:
            return (
                y,
                {
                    'attn_tem_list': attn_tem_list,
                    'g_spa': g_spa,
                    'featuremap_spa_list': featuremap_spa_list,
                    'x_spa_list': x_spa_list,
                    'x_tem_list': _x_list,
                    'tem_emb': tem_emb,
                    'spa_emb': spa_emb,
                }
            )


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
        # x : n,c,v,t
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


if __name__ == '__main__':

    batch_size = 1

    inputs = torch.ones(batch_size, 20, 100)
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
        norm_type='bn',
        act_type='relu',
        xem_projection=0,
        input_position=1,
        input_velocity=1,
        semantic_joint=1,
        semantic_frame=1,
        semantic_class=0,
        semantic_joint_smp=1,
        semantic_joint_fusion=0,
        semantic_frame_fusion=1,
        semantic_frame_location=0,
        # sgcn_g_res_alpha=-1,
        # sgcn_gt_mode=5,
        sgcn_dims=[128, 256, 256],  # [c2, c3, c3],
        sgcn_kernel=1,  # residual connection in GCN
        # sgcn_padding=0,  # residual connection in GCN
        # sgcn_dropout=0.0,  # residual connection in GCN
        # # int for global res, list for individual gcn
        # sgcn_residual=[0, 0, 0],
        # sgcn_prenorm=False,
        sgcn_ffn=1,
        # sgcn_v_kernel=0,
        sgcn_attn_mode=1,
        sgcn_g_kernel=1,
        sgcn_g_proj_dim=[256, 256, 256],  # c3
        # sgcn_g_proj_shared=False,
        # # sgcn_g_weighted=1,
        sgcn_gt_mode=0,

        # sgcn2_g_proj_dim=256,  # c3
        # sgcn2_dims=[256, 256, 256],
        # sgcn2_kernel=1,
        # sgcn2_g_kernel=0,
        # sgcn2_attn_mode=10,
        # gcn_fpn=10,

        # gcn_fpn=9,
        # gcn_fpn_kernel=[3, 5, 7],
        # gcn_fpn_shared=0,
        # # gcn_fpn_output_merge=1,
        # # bifpn_dim=256,
        # # bifpn_layers=1,
        spatial_maxpool=3,
        # temporal_maxpool=1,
        # aspp_rates=None, 345402520
        t_mode=3,
        # t_maxpool_kwargs=None,
        t_mha_kwargs={
            'd_model': [512, 512],
            'nhead': [1, 1],
            'd_head': [512, 512],
            'dim_feedforward': [512, 512],
            'dim_feedforward_output': [512, 512],
            'dropout': 0.2,
            'activation': "relu",
            'num_layers': 2,
            'norm': 'bn',
            'global_norm': False,
            # 'pos_enc': 'abs',
            # 'max_len': 20
        },
        # multi_t=[[], [], [3, 5, 7], [3, 5, 7]],
        # multi_t=[[3, 5, 7], [3, 5, 7], [3, 5, 7]],
        # multi_t_shared=2,
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
