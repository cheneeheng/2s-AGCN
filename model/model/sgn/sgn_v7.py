# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Taken from :
# https://github.com/microsoft/SGN/blob/master/model.py

# Code refractored

# Continue from on sgn_v6


import torch
from torch import nn
from torch import Tensor
from torch.nn import Module as PyTorchModule

import math
from typing import Tuple, Optional, Union, Type, List

from model.module import *
from model.resource.common_ntu import *

from utils.utils import *


T1 = Type[PyTorchModule]
T2 = List[Optional[Type[PyTorchModule]]]


class SGN(PyTorchModule):

    emb_modes = [0, 1, 2, 3, 4, 5, 6, 7]

    g_activation_fn = nn.Softmax
    activation_fn = nn.ReLU
    # dropout_fn = nn.Dropout2d

    c1, c2, c3, c4 = c1, c2, c3, c4

    parts_3points_wholebody = parts_3points_wholebody
    parts_3points_armandhand = parts_3points_armandhand
    parts_2points_interhandandinterfeet = parts_2points_interhandandinterfeet

    def __init__(self,
                 num_class: int = 60,
                 num_point: int = 25,
                 num_segment: int = 20,
                 in_channels: int = 3,
                 bias: int = 1,
                 dropout: float = 0.0,
                 dropout2d: float = 0.0,
                 c_multiplier: Union[int, float, list] = 1,
                 norm_type: str = 'bn',

                 in_position: int = 1,
                 in_velocity: int = 1,
                 in_part: int = 0,
                 in_part_type: int = 0,
                 in_motion: int = 0,

                 xpos_proj: int = 0,
                 xpar_proj: int = 0,

                 sem_part: int = 0,
                 sem_position: int = 1,
                 sem_frame: int = 1,

                 par_pos_fusion: int = 0,
                 sem_par_fusion: int = 0,
                 sem_pos_fusion: int = 0,
                 sem_fra_fusion: int = 1,
                 subject_fusion: int = 1,

                 subject: int = 0,

                 g_part: int = 0,
                 g_proj_shared: bool = False,
                 g_proj_dim: Union[list, int] = c3,  # c3
                 g_residual: List[int] = [0, 0, 0],
                 gcn_t_kernel: int = 1,
                 gcn_dropout: float = 0.0,

                 t_mode: int = 1,
                 t_kernel: int = 3,
                 t_maxpool_kwargs: Optional[dict] = None,
                 aspp: list = None,

                 spatial_maxpool: int = 1,

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
        assert self.norm_type in ['bn', 'ln']
        if self.norm_type == 'bn':
            self.normalization_fn = nn.BatchNorm2d
            self.normalization_fn_1d = nn.BatchNorm1d
        elif self.norm_type == 'ln':
            self.normalization_fn = lambda x: nn.GroupNorm(1, x)
            self.normalization_fn_1d = lambda x: nn.GroupNorm(1, x)

        self.in_position = in_position
        self.in_velocity = in_velocity
        self.in_part = in_part
        self.in_part_type = in_part_type
        self.in_motion = in_motion
        assert self.in_position in self.emb_modes
        assert self.in_velocity in self.emb_modes
        assert self.in_part in self.emb_modes
        assert self.in_part_type in [0, 1, 2]
        assert self.in_motion in self.emb_modes

        if self.in_part_type == 0:
            self.parts_3points = self.parts_3points_wholebody
        elif self.in_part_type == 1:
            self.parts_3points = self.parts_3points_armandhand
        elif self.in_part_type == 2:
            self.parts_3points = self.parts_2points_interhandandinterfeet
        self.parts_len = len(self.parts_3points)
        self.parts_dim = len(self.parts_3points[0])

        self.xpos_proj = xpos_proj
        self.xpar_proj = xpar_proj
        assert self.xpos_proj in self.emb_modes
        assert self.xpar_proj in self.emb_modes

        self.sem_part = sem_part
        self.sem_position = sem_position
        self.sem_frame = sem_frame
        assert self.sem_part in self.emb_modes
        assert self.sem_position in self.emb_modes
        assert self.sem_frame in self.emb_modes
        if self.in_position == 0 and self.sem_position > 0:
            raise ValueError("in_position is 0 but sem_position is not")
        if self.in_part == 0 and self.sem_part > 0:
            raise ValueError("in_part is 0 but sem_part is not")

        self.par_pos_fusion = par_pos_fusion
        self.sem_pos_fusion = sem_pos_fusion
        self.sem_par_fusion = sem_par_fusion
        self.sem_fra_fusion = sem_fra_fusion
        self.subject_fusion = subject_fusion
        # 0 = concat before sgn, 1 = concat after sgn, others have projection
        assert self.par_pos_fusion in [0, 1, 2, 3, 4, 5]
        # 0 = concat, 1 = add
        assert self.sem_pos_fusion in [0, 1]
        assert self.sem_par_fusion in [0, 1]
        # 1 = add after GCN, 101 = add before GCN
        assert self.sem_fra_fusion in [1, 101]
        assert self.subject_fusion in [1, 101]

        self.subject = subject
        assert self.subject in [0, 1, 2, 3, 4]

        self.g_part = g_part
        assert self.g_part in [0, 1, 2, 3]
        self.g_proj_shared = g_proj_shared
        self.g_proj_dim = g_proj_dim
        self.g_residual = g_residual
        self.gcn_t_kernel = gcn_t_kernel
        self.gcn_dropout_fn = lambda: nn.Dropout2d(gcn_dropout)

        if self.sem_pos_fusion == 1 or self.sem_par_fusion == 1:
            self.gcn_in_ch = self.c1
        elif self.sem_position > 0 or self.sem_part > 0:
            self.gcn_in_ch = self.c1 * 2
        else:
            self.gcn_in_ch = self.c1

        self.t_mode = t_mode
        self.t_kernel = t_kernel
        assert self.t_kernel >= 0
        self.t_maxpool_kwargs = t_maxpool_kwargs
        self.aspp = aspp

        self.spatial_maxpool = spatial_maxpool

        # Dynamic Representation -----------------------------------------------
        self.init_input_dr()

        # Joint Embedding ------------------------------------------------------
        # Group Embedding ------------------------------------------------------
        # Frame Embedding ------------------------------------------------------
        self.init_input_sem()

        if self.in_position > 0 or self.in_velocity > 0:
            if self.xpos_proj > 0:
                if self.sem_pos_fusion == 1:
                    in_channels = self.c1
                else:
                    in_channels = self.c1*2
                inter_channels = self.get_inter_channels(self.xpos_proj,
                                                         self.c2)
                self.xpos_projection = self.init_emb(
                    mode=self.xpos_proj,
                    num_point=self.num_point,
                    in_channels=in_channels,
                    out_channels=self.c2,
                    inter_channels=inter_channels,
                )

        if self.in_part > 0 or self.in_motion > 0:
            if self.xpar_proj > 0:
                if self.sem_pos_fusion == 1:
                    in_channels = self.c1
                else:
                    in_channels = self.c1*2
                self.xpar_projection = self.init_emb(
                    mode=self.xpar_proj,
                    num_point=self.parts_len,
                    in_channels=in_channels,
                    out_channels=self.c2,
                    inter_channels=self.c2,
                )

        # Subject Embedding ----------------------------------------------------
        self.init_input_subject()

        # Position Embedding ---------------------------------------------------
        # Frame embedding is a form of PE

        # GCN ------------------------------------------------------------------
        self.init_spatial_gcn()
        self.init_spatial_fusion()

        # ASPP -----------------------------------------------------------------
        # Frame level module ---------------------------------------------------
        self.init_temporal_mlp()

        if self.spatial_maxpool == 1:
            self.smp = nn.AdaptiveMaxPool2d((1, num_segment))

        self.tmp = nn.AdaptiveMaxPool2d((1, 1))

        self.do = nn.Dropout(dropout)

        if self.t_mode == 0:
            self.fc = nn.Linear(self.c3, num_class)
        else:
            self.fc = nn.Linear(self.c4, num_class)

        self.init_weight()

        if self.in_part > 0 or self.in_motion > 0:
            self.register_buffer(
                'parts_3points_vec',
                torch.tensor(self.parts_3points, dtype=torch.int).reshape(-1)
            )

    def get_inter_channels(self, mode: int, ch: int) -> Union[list, int]:
        if mode == 3:
            return [ch, ch, ch]
        elif mode == 7:
            return []
        elif mode == 5:
            # inspired by the 4x dim in ffn of transformers
            return ch * 4
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

        def _init_0(x: Tensor): return nn.init.constant_(x, 0)

        _init_0(self.gcn_spatial.gcn1.w1.block.conv.conv.weight)
        _init_0(self.gcn_spatial.gcn2.w1.block.conv.conv.weight)
        _init_0(self.gcn_spatial.gcn3.w1.block.conv.conv.weight)
        if self.g_part == 0:
            _init_0(self.gcn_spatial_part.gcn1.w1.block.conv.conv.weight)
            _init_0(self.gcn_spatial_part.gcn2.w1.block.conv.conv.weight)
            _init_0(self.gcn_spatial_part.gcn3.w1.block.conv.conv.weight)

    def init_input_dr(self):
        if self.in_position > 0:
            self.pos_embed = self.init_emb(
                mode=self.in_position,
                num_point=self.num_point,
                in_channels=self.in_channels,
                in_norm=self.normalization_fn_1d
            )
        if self.in_velocity > 0:
            self.vel_embed = self.init_emb(
                mode=self.in_velocity,
                num_point=self.num_point,
                in_channels=self.in_channels,
                in_norm=self.normalization_fn_1d
            )
        if self.in_part > 0:
            self.par_embed = self.init_emb(
                mode=self.in_part,
                num_point=self.parts_len,
                in_channels=self.in_channels*self.parts_dim,
                in_norm=self.normalization_fn_1d
            )
        if self.in_motion > 0:
            self.mot_embed = self.init_emb(
                mode=self.in_motion,
                num_point=self.parts_len,
                in_channels=self.in_channels*self.parts_dim,
                in_norm=self.normalization_fn_1d
            )

    def init_input_subject(self):
        if self.subject > 0:
            inter_channels = self.get_inter_channels(self.subject, self.c1)
            if self.subject_fusion == 1:
                out_channels = self.c3
            elif self.subject_fusion == 101:
                out_channels = self.gcn_in_ch
            self.sub_embed = EmbeddingSubject(
                self.c1,
                out_channels,
                bias=self.bias,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                inter_channels=inter_channels,
                num_subjects=2,
                mode=self.subject
            )

    def init_input_sem(self):
        # Joint Embedding
        if self.sem_position > 0:
            self.spa = OneHotTensor(self.num_point, self.num_segment, mode=0)
            self.spa_embed = self.init_emb(mode=self.sem_position,
                                           num_point=self.num_point,
                                           in_channels=self.num_point)
        # Group Embedding
        if self.sem_part > 0:
            self.gro = OneHotTensor(self.parts_len, self.num_segment, mode=0)
            self.gro_embed = self.init_emb(mode=self.sem_part,
                                           num_point=self.parts_len,
                                           in_channels=self.parts_len)
        # Frame Embedding
        if self.sem_frame > 0:
            if self.in_position == 0 and self.in_velocity == 0:
                if self.in_part > 0 or self.in_motion > 0:
                    self.tem = OneHotTensor(self.num_segment,
                                            self.parts_len,
                                            mode=1)
            else:
                if self.in_part > 0 or self.in_motion > 0:
                    self.tem = OneHotTensor(self.num_segment,
                                            self.num_point+self.parts_len,
                                            mode=1)
                else:
                    self.tem = OneHotTensor(self.num_segment,
                                            self.num_point,
                                            mode=1)
            if self.sem_fra_fusion == 1:
                out_channels = self.c3
            elif self.sem_fra_fusion == 101:
                out_channels = self.gcn_in_ch
            self.tem_embed = self.init_emb(mode=self.sem_frame,
                                           num_point=self.num_point,
                                           in_channels=self.num_segment,
                                           out_channels=out_channels)

    def init_spatial_gcn(self):
        self.gcn_spatial = GCNSpatialBlock(
            0,
            0,
            kernel_size=self.gcn_t_kernel,
            padding=self.gcn_t_kernel//2,
            bias=self.bias,
            dropout=self.gcn_dropout_fn,
            activation=self.activation_fn,
            normalization=self.normalization_fn,
            gcn_dims=[self.gcn_in_ch, self.c2, self.c3, self.c3],
            g_proj_dim=self.g_proj_dim,
            g_proj_shared=self.g_proj_shared,
            g_activation=self.g_activation_fn,
            g_residual=self.g_residual
        )
        if self.g_part == 0:
            self.gcn_spatial_part = GCNSpatialBlock(
                0,
                0,
                kernel_size=self.gcn_t_kernel,
                padding=self.gcn_t_kernel//2,
                bias=self.bias,
                dropout=self.gcn_dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                gcn_dims=[self.gcn_in_ch, self.c2, self.c3, self.c3],
                g_proj_dim=self.g_proj_dim,
                g_proj_shared=self.g_proj_shared,
                g_activation=self.g_activation_fn,
                g_residual=self.g_residual
            )
        elif self.g_part > 0 and self.par_pos_fusion % 2 == 1:   # post fusion
            in_channels = self.c2
            out_channels = self.c3
            self.non_gcn_proj = self.init_emb(mode=self.g_part,
                                              num_point=self.num_point,
                                              in_channels=in_channels,
                                              out_channels=out_channels)

    def init_spatial_fusion(self):
        if self.par_pos_fusion in [0, 2, 4]:
            in_channels = self.c2
            out_channels = self.c2
        elif self.par_pos_fusion in [1, 3, 5]:
            in_channels = self.c3
            out_channels = self.c3
        self.fuse_spatial = SpatialFusion(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=self.bias,
            dropout=self.dropout_fn,
            activation=self.activation_fn,
            normalization=self.normalization_fn,
            mode=self.par_pos_fusion
        )

    def init_temporal_mlp(self):
        if self.aspp is None or len(self.aspp) == 0:
            self.aspp = lambda x: x
        else:
            self.aspp = ASPP(self.c3,
                             self.c3,
                             bias=self.bias,
                             dilation=self.aspp,
                             dropout=self.dropout_fn,
                             activation=self.activation,
                             normalization=self.normalization_fn)
        # skip
        if self.t_mode == 0:
            self.cnn = lambda x: x
        # original sgn
        elif self.t_mode == 1:
            idx = 2
            channels = [self.c3, self.c3, self.c4]
            kernel_sizes = [self.t_kernel, 1]
            paddings = [self.t_kernel//2, 0]
            residuals = [0 for _ in range(idx)]
            dropouts = [self.dropout_fn, None]
        # original sgn with residual
        elif self.t_mode == 2:
            idx = 2
            channels = [self.c3, self.c3, self.c4]
            kernel_sizes = [self.t_kernel, 1]
            paddings = [self.t_kernel//2, 0]
            residuals = [1 for _ in range(idx)]
            dropouts = [self.dropout_fn, None]
        # all 3x3
        elif self.t_mode == 3:
            idx = 2
            channels = [self.c3, self.c3, self.c4]
            kernel_sizes = [self.t_kernel for _ in range(2)]
            paddings = [self.t_kernel//2 for _ in range(2)]
            residuals = [0 for _ in range(2)]
            dropouts = [self.dropout_fn, None]
        # all 3x3 with residual
        elif self.t_mode == 4:
            idx = 2
            channels = [self.c3, self.c3, self.c4]
            kernel_sizes = [self.t_kernel for _ in range(2)]
            paddings = [self.t_kernel//2 for _ in range(2)]
            residuals = [1 for _ in range(2)]
            dropouts = [self.dropout_fn, None]
        # 3 layers
        elif self.t_mode == 5:
            idx = 3
            channels = [self.c3, self.c3, self.c3, self.c4]
            kernel_sizes = [self.t_kernel, 1, 1]
            paddings = [self.t_kernel//2, 0, 0]
            residuals = [0 for _ in range(3)]
            dropouts = [self.dropout_fn, None, None]
        # 3 layers with residual
        elif self.t_mode == 6:
            idx = 3
            channels = [self.c3, self.c3, self.c3, self.c4]
            kernel_sizes = [self.t_kernel, 1, 1]
            paddings = [self.t_kernel//2, 0, 0]
            residuals = [1 for _ in range(3)]
            dropouts = [self.dropout_fn, None, None]
        # original sgn + all dropout
        elif self.t_mode == 7:
            idx = 2
            channels = [self.c3, self.c3, self.c4]
            kernel_sizes = [self.t_kernel, 1]
            paddings = [self.t_kernel//2, 0]
            residuals = [0 for _ in range(idx)]
            dropouts = [self.dropout_fn, self.dropout_fn]
        # original sgn + all dropout + residual
        elif self.t_mode == 8:
            idx = 2
            channels = [self.c3, self.c3, self.c4]
            kernel_sizes = [self.t_kernel, 1]
            paddings = [self.t_kernel//2, 0]
            residuals = [1 for _ in range(idx)]
            dropouts = [self.dropout_fn, self.dropout_fn]
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
            maxpool_kwargs=self.t_maxpool_kwargs
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

    def get_dy2(self, x: Tensor):
        bs, step, dim = x.shape
        num_point = dim // 3
        x1 = x.view((bs, step, num_point, 3))  # n,t,v,c
        if self.in_part > 0:
            par = torch.index_select(x1, 2, self.parts_3points_vec)  # n,t,v+,c
            par = par.view(
                (bs, step, -1, len(self.parts_3points[0]), self.in_channels)
            )  # n,t,v+,3,c
            mid = par.mean(dim=-2, keepdim=True)  # n,t,v+,1,c
            par1 = par - mid  # n,t,v+,3,c
            par = par1.view(
                (bs, step, -1, self.in_channels*len(self.parts_3points[0]))
            )  # n,t,v+,c+
            par = par.permute(0, 3, 2, 1).contiguous()  # n,c+,v+,t
            par = self.par_embed(par)  # n,c,v+,t

        if self.in_motion > 0:
            if self.in_part == 0:
                par = torch.index_select(x1, 2, self.parts_3points_vec)
                par = par.view((bs, step, -1,
                                len(self.parts_3points[0]), self.in_channels))
                mid = par.mean(dim=-2, keepdim=True)
                par1 = par - mid
            mot = par1[:, 1:] - par1[:, :-1]  # n,t-1,v+,3,c
            mot = mot.view((*mot.shape[:3], -1))  # n,t-1,v+,c+
            mot = mot.permute(0, 3, 2, 1).contiguous()  # n,c,v+,t-1
            mot = self.pad_zeros(mot)  # n,c,v+,t
            mot = self.mot_embed(mot)

        if self.in_part > 0 and self.in_motion > 0:
            dy2 = par + mot
        elif self.in_part > 0:
            dy2 = par
        elif self.in_motion > 0:
            dy2 = mot
        else:
            dy2 = None

        return dy2

    def get_emb(self,
                x: Tensor,
                s: Optional[Tensor] = None
                ) -> Tuple[Optional[Tensor],
                           Optional[Tensor],
                           Optional[Tensor],
                           Optional[Tensor]]:
        spa1, gro1, tem1, sub1 = None, None, None, None
        bs, step, dim = x.shape
        if self.sem_position > 0:
            spa1 = self.spa_embed(self.spa(bs))
        if self.sem_part > 0:
            gro1 = self.gro_embed(self.gro(bs))
        if self.sem_frame > 0:
            tem1 = self.tem_embed(self.tem(bs))
        if self.subject > 0:
            s = s.view((bs, step, 1, 1))  # n,t,v,c
            s = s.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
            sub1 = self.sub_embed(s)
        return spa1, gro1, tem1, sub1

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
        dy2 = self.get_dy2(x)

        assert dy1 is not None or dy2 is not None

        # Joint and frame embeddings -------------------------------------------
        spa1, gro1, tem1, sub1 = self.get_emb(x, s)

        # Joint-level Module ---------------------------------------------------
        # xyz embeddings
        if dy1 is None:
            x_pos = None
        else:
            if spa1 is None:
                x_pos = dy1  # n,c,v,t
            else:
                if self.sem_pos_fusion == 1:
                    x_pos = dy1 + spa1  # n,c,v,t
                else:
                    x_pos = torch.cat([dy1, spa1], 1)  # n,c,v,t

        # parts embeddings
        if dy2 is None:
            x_par = None
        else:
            if gro1 is None:
                x_par = dy2  # n,c,v',t
            else:
                if self.sem_par_fusion == 1:
                    x_par = dy2 + gro1  # n,c,v',t
                else:
                    x_par = torch.cat([dy2, gro1], 1)  # n,c,v',t

        if hasattr(self, 'xpos_projection'):
            x_pos = self.xpos_projection(x_pos)
        if hasattr(self, 'xpar_projection'):
            x_par = self.xpar_projection(x_par)

        # spatial fusion pre gcn
        x, fusion_level = self.fuse_spatial(x1=x_pos, x2=x_par)

        # temporal fusion pre gcn
        if self.sem_frame > 0 and self.sem_fra_fusion == 101:
            x = [i + tem1 for i in x]
        if self.subject > 0 and self.subject_fusion == 101:
            x = [i + sub1 for i in x]

        # GCN ------------------------------------------------------------------
        x0, g0 = self.gcn_spatial(x[0])

        if self.par_pos_fusion % 2 == 1:   # post fusion
            if self.g_part == 0:
                x1, g1 = self.gcn_spatial_part(x[1])
            elif self.g_part > 0:
                x1 = self.non_gcn_proj(x[1])
                g1 = None
            x, g = [x0, x1], [g0, g1]
        else:
            x, g = [x0], [g0]

        # Frame-level Module ---------------------------------------------------
        # spatial fusion post gcn
        x, _ = self.fuse_spatial(*x, fusion_level=fusion_level)
        x = x[0]

        # temporal fusion post gcn
        if self.sem_frame > 0 and self.sem_fra_fusion == 1:
            if self.par_pos_fusion % 2 == 0:
                x = x + tem1
        if self.subject > 0 and self.subject_fusion == 1:
            if self.par_pos_fusion % 2 == 0:
                x = x + sub1

        if self.spatial_maxpool == 1:
            x = self.smp(x)
        x = self.aspp(x)
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
        bs, _, num_point, step = x.shape
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
        assert mode in [1, 2, 3, 4, 5, 6, 7]
        self.mode = mode

        if in_norm is not None:
            self.norm = DataNorm(self.in_channels * num_point, in_norm)
        else:
            self.norm = lambda x: x

        if self.mode in [1, 4, 5, 6]:
            # 1=original, 4=dropout, 5=higher inter, 6=residual
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
                    self.res1 = lambda x: x
                else:
                    self.res1 = Conv(self.in_channels, inter_channels,
                                     bias=self.bias)
                if inter_channels == self.out_channels:
                    self.res2 = lambda x: x
                else:
                    self.res2 = Conv(inter_channels, self.out_channels,
                                     bias=self.bias)
            else:
                self.res1 = lambda x: 0
                self.res2 = lambda x: 0

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
        if self.mode in [1, 4, 5, 6]:
            x = self.cnn1(x) + self.res1(x)
            x = self.cnn2(x) + self.res2(x)
        elif self.mode == 2:
            x = self.cnn1(x)
        elif self.mode in [3, 7]:
            for i in range(self.num_layers):
                x = getattr(self, f'cnn{i+1}')(x)
        return x


class EmbeddingSubject(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: int = 0,
                 dropout: T1 = nn.Dropout2d,
                 activation: T1 = nn.ReLU,
                 normalization: T1 = nn.BatchNorm2d,
                 inter_channels: Union[list, int] = 0,
                 num_subjects: int = 2,
                 mode: int = 1,
                 ):
        super(EmbeddingSubject, self).__init__(in_channels,
                                               out_channels,
                                               bias=bias,
                                               dropout=dropout,
                                               activation=activation,
                                               normalization=normalization)
        assert mode in [1, 2, 3, 4]
        self.mode = mode
        if mode == 1:
            # 2x MLP
            self.in_dim = self.in_channels
            self.cnn1 = Conv(self.in_channels,
                             inter_channels,
                             bias=self.bias,
                             activation=self.activation)
            self.cnn2 = Conv(inter_channels,
                             self.out_channels,
                             bias=self.bias,
                             activation=self.activation)
        elif mode == 2:
            # bert style
            self.in_dim = self.out_channels
            embedding = torch.empty(num_subjects, self.out_channels)
            nn.init.normal_(embedding, std=0.02)  # bert
            self.embedding = nn.Parameter(embedding)
            self.norm = self.normalization(self.out_channels)
            self.drop = self.dropout()
        elif mode == 3:
            # 4x MLP
            self.in_dim = self.in_channels
            assert isinstance(inter_channels, list)
            layer_channels = \
                [self.in_channels] + inter_channels + [self.out_channels]
            self.num_layers = len(layer_channels)-1
            for i in range(self.num_layers):
                setattr(self, f'cnn{i+1}', Conv(layer_channels[i],
                                                layer_channels[i+1],
                                                bias=self.bias,
                                                activation=self.activation))
        elif mode == 4:
            # bert style and a conv
            self.in_dim = self.in_channels
            embedding = torch.empty(num_subjects, self.in_channels)
            nn.init.normal_(embedding, std=0.02)  # bert
            self.embedding = nn.Parameter(embedding)
            self.cnn1 = Conv(self.in_channels,
                             self.out_channels,
                             bias=self.bias,
                             activation=self.activation)

    def forward(self, x: Tensor) -> Tensor:
        bs, _, _, step = x.shape  # n,c,v,t => n,1,1,t
        x = x.reshape(-1).unsqueeze(-1)  # nt,1
        x = x.expand(x.shape[0], self.in_dim)
        if hasattr(self, "embedding"):
            x = torch.gather(self.embedding, 0, x.type(torch.int64))
        x = x.reshape((bs, step, 1, self.in_dim))
        x = x.transpose(1, -1)  # n,c,v,t
        if self.mode == 1:
            x = self.cnn1(x)
            x = self.cnn2(x)
        elif self.mode == 2:
            x = self.norm(x)
            x = self.drop(x)
        elif self.mode == 3:
            for i in range(self.num_layers):
                x = getattr(self, f'cnn{i+1}')(x)
        elif self.mode == 4:
            x = self.cnn1(x)
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
                 maxpool_kwargs: Optional[dict] = None
                 ):
        super(MLPTemporal, self).__init__()
        if maxpool_kwargs is not None:
            # self.pool = nn.MaxPool2d(kernel_size=(1, self.kernel_size),
            #                          stride=(1, max_pool_stride))
            self.pool = nn.MaxPool2d(**maxpool_kwargs)
        else:
            self.pool = lambda x: x
        self.num_layers = len(channels) - 1
        for i in range(self.num_layers):
            setattr(self,
                    f'cnn{i+1}',
                    Conv(channels[i],
                         channels[i+1],
                         kernel_size=kernel_sizes[i],
                         padding=paddings[i],
                         bias=biases[i],
                         activation=activations[i],
                         normalization=lambda: normalizations[i](
                             channels[i+1]),
                         dropout=dropouts[i])
                    )
            if residuals[i] == 0:
                setattr(self, f'res{i+1}', lambda x: 0)
            elif residuals[i] == 1:
                if channels[i] == channels[i+1]:
                    setattr(self, f'res{i+1}', lambda x: x)
                else:
                    setattr(self, f'res{i+1}', Conv(channels[i], channels[i+1],
                                                    bias=biases[i]))
            else:
                raise ValueError('Unknown residual mode...')

    def forward(self, x: Tensor) -> Tensor:
        # x: n,c,v,t ; v=1 due to SMP
        x = self.pool(x)
        for i in range(self.num_layers):
            x = getattr(self, f'cnn{i+1}')(x) + getattr(self, f'res{i+1}')(x)
        return x


class GCNSpatialG(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: int = 0,
                 activation: T1 = nn.Softmax,
                 g_proj_shared: bool = False,
                 ):
        super(GCNSpatialG, self).__init__(in_channels,
                                          out_channels,
                                          bias=bias,
                                          activation=activation)
        self.g1 = Conv(self.in_channels, self.out_channels, bias=self.bias)
        if g_proj_shared:
            self.g2 = self.g1
        else:
            self.g2 = Conv(self.in_channels, self.out_channels, bias=self.bias)
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
                 normalization: T1 = nn.BatchNorm2d
                 ):
        super(GCNSpatialUnit, self).__init__(in_channels,
                                             out_channels,
                                             kernel_size=kernel_size,
                                             padding=padding,
                                             bias=bias,
                                             dropout=dropout,
                                             activation=activation,
                                             normalization=normalization)
        self.norm = self.normalization(self.out_channels)
        self.act = self.activation()
        self.drop = (lambda x: x) if self.dropout is None else self.dropout()
        self.w1 = Conv(self.in_channels, self.out_channels, bias=self.bias)
        self.w2 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                       kernel_size=self.kernel_size, padding=self.padding)

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        x1 = x.permute(0, 3, 2, 1).contiguous()  # n,t,v,c
        x1 = g.matmul(x1)
        x1 = x1.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
        x1 = self.w1(x1) + self.w2(x)  # z + residual
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
                 gcn_dims: list,
                 g_proj_dim: Tuple[int, list],
                 g_proj_shared: bool = False,
                 g_activation: T1 = nn.Softmax,
                 g_residual: List[int] = [0, 0, 0],
                 ):
        super(GCNSpatialBlock, self).__init__(*args,
                                              kernel_size=kernel_size,
                                              padding=padding,
                                              bias=bias,
                                              dropout=dropout,
                                              activation=activation,
                                              normalization=normalization)
        self.g_shared = isinstance(g_proj_dim, int)
        if self.g_shared:
            self.gcn_g = GCNSpatialG(gcn_dims[0],
                                     g_proj_dim,
                                     bias=self.bias,
                                     activation=g_activation,
                                     g_proj_shared=g_proj_shared)
        else:
            self.gcn_g1 = GCNSpatialG(gcn_dims[0],
                                      g_proj_dim[0],
                                      bias=self.bias,
                                      activation=g_activation,
                                      g_proj_shared=g_proj_shared)
            self.gcn_g2 = GCNSpatialG(gcn_dims[1],
                                      g_proj_dim[1],
                                      bias=self.bias,
                                      activation=g_activation,
                                      g_proj_shared=g_proj_shared)
            self.gcn_g3 = GCNSpatialG(gcn_dims[2],
                                      g_proj_dim[2],
                                      bias=self.bias,
                                      activation=g_activation,
                                      g_proj_shared=g_proj_shared)

        self.gcn1 = GCNSpatialUnit(gcn_dims[0],
                                   gcn_dims[1],
                                   bias=self.bias,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   dropout=self.dropout,
                                   activation=self.activation,
                                   normalization=self.normalization)
        self.gcn2 = GCNSpatialUnit(gcn_dims[1],
                                   gcn_dims[2],
                                   bias=self.bias,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   dropout=self.dropout,
                                   activation=self.activation,
                                   normalization=self.normalization)
        self.gcn3 = GCNSpatialUnit(gcn_dims[2],
                                   gcn_dims[3],
                                   bias=self.bias,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   dropout=self.dropout,
                                   activation=self.activation,
                                   normalization=self.normalization)

        for i, r in enumerate(g_residual):
            if r == 0:
                setattr(self, f'res{i+1}', lambda x: 0)
            elif r == 1:
                if gcn_dims[i] == gcn_dims[i+1]:
                    setattr(self, f'res{i+1}', lambda x: x)
                else:
                    setattr(self, f'res{i+1}', Conv(gcn_dims[i], gcn_dims[i+1],
                                                    bias=self.bias))
            else:
                raise ValueError("Unknown residual modes...")

    def forward(self, x: Tensor) -> Tensor:
        if self.g_shared:
            g = self.gcn_g(x)
            x = self.gcn1(x, g) + self.res1(x)
            x = self.gcn2(x, g) + self.res2(x)
            x = self.gcn3(x, g) + self.res3(x)
        else:
            g1 = self.gcn_g1(x)
            x = self.gcn1(x, g1) + self.res1(x)
            g2 = self.gcn_g2(x)
            x = self.gcn2(x, g2) + self.res1(x)
            g3 = self.gcn_g3(x)
            x = self.gcn3(x, g3) + self.res1(x)
            g = [g1, g2, g3]
        return x, g


class SpatialFusion(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: int = 0,
                 dropout: T1 = nn.Dropout2d,
                 activation: T1 = nn.ReLU,
                 normalization: T1 = nn.BatchNorm2d,
                 mode: int = 1
                 ):
        super(SpatialFusion, self).__init__(in_channels,
                                            out_channels,
                                            bias=bias,
                                            dropout=dropout,
                                            activation=activation,
                                            normalization=normalization)
        assert mode in [0, 1, 2, 3, 4, 5]
        self.mode = mode
        if self.mode == 2 or self.mode == 3:
            self.cnn1 = Conv(self.in_channels,
                             self.out_channels,
                             bias=self.bias)
        elif self.mode == 4 or self.mode == 5:
            self.cnn1 = Conv(self.in_channels,
                             self.in_channels,
                             bias=self.bias,
                             activation=self.activation)
            self.cnn2 = Conv(self.in_channels,
                             self.out_channels,
                             bias=self.bias)

    def forward(self,
                x1: Optional[Union[Tensor, list]] = None,
                x2: Optional[Tensor] = None,
                fusion_level: int = 0) -> Tuple[list, bool]:
        """Fusion of features from x1 (x_pos) and x2 (x_par).

        Args:
            x1 (Optional[Union[Tensor, list]]): x_pos
            x2 (Optional[Tensor]): x_par
            fusion_level (int): which stage of the fusion.

        Returns:
            Tensor: fused tensor or a list or original tensors.
        """
        fuse_flag = False
        if fusion_level == 0:
            if self.mode in [1, 3, 5]:
                assert x1 is not None and x2 is not None
                x = [x1, x2]
            elif x1 is not None and x2 is not None:
                x = torch.cat([x1, x2], 2)  # n,c,v'+v,t
                fuse_flag = True
            elif x1 is not None:
                x = x1
            elif x2 is not None:
                x = x2
            else:
                raise ValueError("Unsupported input combination")

        elif fusion_level == 1:
            if self.mode in [0, 2, 4]:
                assert x2 is None
                x = x1
            else:
                assert x2 is not None and x2 is not None
                x = torch.cat([x1, x2], 2)  # n,c,v'+v,t
                fuse_flag = True

        # projection
        if fuse_flag:
            if self.mode == 2 or self.mode == 3:
                if fusion_level:
                    x = self.cnn1(x)
            elif self.mode == 4 or self.mode == 5:
                if fusion_level:
                    x = self.cnn1(x)
                    x = self.cnn2(x)

        if not isinstance(x, list):
            x = [x]

        fusion_level += 1
        return x, fusion_level


if __name__ == '__main__':

    batch_size = 64

    inputs = torch.ones(batch_size, 20, 75)
    subjects = torch.ones(batch_size, 20, 1)

    model = SGN(num_segment=20,
                in_position=1,
                in_velocity=1,
                # in_part=1,
                # in_motion=1,
                # in_part_type=2,
                xpos_proj=7,
                # par_pos_fusion=3,
                # # # subject=1,
                # sem_part=1,
                # g_part=0,
                # # sem_fra_fusion=1,
                # # subject_fusion=101
                # c_multiplier=[1, 0.5, 0.25, 0.125],
                # t_mode=7,
                # gcn_dropout=0.2,
                )
    model(inputs, subjects)
    # print(model)
