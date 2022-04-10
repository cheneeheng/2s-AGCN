# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Taken from :
# https://github.com/microsoft/SGN/blob/master/model.py

# Code refractored

# Continue from on sgn_v5

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module as PyTorchModule
from torch.profiler import profile, ProfilerActivity

import math
from typing import Tuple, Optional, Union, Type

from model.module.pytorch_module_wrapper import *
from model.module.pos_embedding import *
from model.resource.common_ntu import *

from utils.utils import *


class SGN(PyTorchModule):

    c1 = 64  # pos,vel,joint embed
    c2 = 128  # G,gcn
    c3 = 256  # gcn
    c4 = 512  # final conv

    parts_3points_wholebody = parts_3points_wholebody
    parts_3points_armandhand = parts_3points_armandhand
    parts_2points_interhandandinterfeet = parts_2points_interhandandinterfeet

    def __init__(self,
                 num_class: int = 60,
                 num_point: int = 25,
                 num_segment: int = 20,
                 in_channels: int = 3,
                 bias: bool = True,
                 dropout: float = 0.0,
                 c_multiplier: Union[int, float] = 1,

                 in_position: int = 1,
                 in_velocity: int = 1,
                 in_part: int = 0,
                 in_part_type: int = 0,
                 in_motion: int = 0,

                 in_par_mot_fusion: int = 0,
                 in_pos_vel_fusion: int = 0,

                 sem_part: int = 0,
                 sem_position: int = 1,
                 sem_frame: int = 1,

                 subject: int = 0,

                 g_shared: bool = True,
                 g_proj_shared: bool = False,
                 g_proj_dim: Union[list, int] = c3,  # c3
                 gcn_t_kernel: int = 1,

                 t_kernel: int = 3,
                 t_max_pool: int = 0,
                 aspp: list = None,

                 norm_type: str = 'bn'
                 ):
        super(SGN, self).__init__()

        self.c1 = to_int(self.c1 * c_multiplier)  # pos,vel,joint embed
        self.c2 = to_int(self.c2 * c_multiplier)  # G,gcn
        self.c3 = to_int(self.c3 * c_multiplier)  # gcn
        self.c4 = to_int(self.c4 * c_multiplier)  # gcn

        self.num_class = num_class
        self.num_point = num_point
        self.num_segment = num_segment
        self.in_channels = in_channels
        self.bias = bias

        self.in_position = in_position
        self.in_velocity = in_velocity
        self.in_part = in_part
        self.in_part_type = in_part_type
        self.in_motion = in_motion

        self.in_pos_vel_fusion = in_pos_vel_fusion
        self.in_par_mot_fusion = in_par_mot_fusion

        self.sem_part = sem_part
        self.sem_position = sem_position
        self.sem_frame = sem_frame

        self.subject = subject

        self.g_shared = g_shared
        self.g_proj_shared = g_proj_shared
        self.g_proj_dim = g_proj_dim
        self.gcn_t_kernel = gcn_t_kernel

        if not self.g_shared and not isinstance(self.g_proj_dim, list):
            self.g_proj_dim = [self.g_proj_dim] * 3

        self.t_kernel = t_kernel
        self.t_max_pool = t_max_pool

        self.norm_type = norm_type

        assert self.subject in [0, 1, 2, 3, 4]

        dr_modes = [0, 1, 2, 3]
        assert self.in_position in dr_modes
        assert self.in_velocity in dr_modes
        assert self.in_part in dr_modes
        assert self.in_part_type in [0, 1, 2]
        assert self.in_motion in dr_modes

        # 0 = concat, 1 = add
        assert self.in_pos_vel_fusion in [0, 1]
        assert self.in_par_mot_fusion in [0, 1]

        sem_modes = [0, 1, 2, 3]
        assert self.sem_part in sem_modes
        assert self.sem_position in sem_modes
        assert self.sem_frame in sem_modes

        if self.in_position == 0 and self.sem_position > 0:
            raise ValueError("in_position is 0 but sem_position is not")
        if self.in_part == 0 and self.sem_part > 0:
            raise ValueError("in_part is 0 but sem_part is not")

        assert self.t_kernel > 0
        assert self.t_max_pool >= 0
        assert self.norm_type in ['bn', 'ln']

        if self.norm_type == 'bn':
            self.norm_mod = nn.BatchNorm2d
            self.norm_mod_1d = nn.BatchNorm1d
        elif self.norm_type == 'ln':
            self.norm_mod = lambda x: nn.GroupNorm(1, x)
            self.norm_mod_1d = lambda x: nn.GroupNorm(1, x)

        if in_part_type == 0:
            self.parts_3points = self.parts_3points_wholebody
        elif in_part_type == 1:
            self.parts_3points = self.parts_3points_armandhand
        elif in_part_type == 2:
            self.parts_3points = self.parts_2points_interhandandinterfeet

        parts_len = len(self.parts_3points)
        parts_dim = len(self.parts_3points[0])

        # Dynamic Representation -----------------------------------------------
        if self.in_position > 0:
            self.pos_embed = self.init_dr(mode=self.in_position,
                                          num_point=num_point,
                                          in_channels=in_channels)

        if self.in_velocity > 0:
            self.vel_embed = self.init_dr(mode=self.in_velocity,
                                          num_point=num_point,
                                          in_channels=in_channels)

        if self.in_part > 0:
            self.par_embed = self.init_dr(mode=self.in_part,
                                          num_point=parts_len,
                                          in_channels=in_channels*parts_dim)

        if self.in_motion > 0:
            self.mot_embed = self.init_dr(mode=self.in_motion,
                                          num_point=parts_len,
                                          in_channels=in_channels*parts_dim)

        # Joint Embedding ------------------------------------------------------
        if self.sem_position > 0:
            self.spa = OneHotTensor(num_point, num_segment, mode=0)
            self.spa_embed = self.init_emb(mode=self.sem_position,
                                           num_point=num_point,
                                           in_channels=num_point)

        # Group Embedding ------------------------------------------------------
        if self.sem_part > 0:
            self.gro = OneHotTensor(parts_len, num_segment, mode=0)
            self.gro_embed = self.init_emb(mode=self.sem_part,
                                           num_point=parts_len,
                                           in_channels=parts_len)

        # Frame Embedding ------------------------------------------------------
        if self.sem_frame > 0:
            if self.in_position == 0 and self.in_velocity == 0:
                if self.in_part > 0 or self.in_motion > 0:
                    self.tem = OneHotTensor(num_segment, parts_len, mode=1)  # noqa
            else:
                if self.in_part > 0 or self.in_motion > 0:
                    self.tem = OneHotTensor(num_segment, num_point+parts_len, mode=1)  # noqa
                else:
                    self.tem = OneHotTensor(num_segment, num_point, mode=1)

            self.tem_embed = self.init_emb(mode=self.sem_frame,
                                           num_point=num_point,
                                           in_channels=num_segment,
                                           out_channels=self.c3)

        # Subject Embedding ----------------------------------------------------
        if self.subject > 0:
            inter_channels = self.get_inter_channels(self.subject, self.c1)
            self.sub_embed = EmbeddingSubject(self.c1,
                                              self.c3,
                                              inter_channels=inter_channels,
                                              num_subjects=2,
                                              bias=bias,
                                              norm_mod=self.norm_mod,
                                              mode=self.subject)

        # Position Embedding ---------------------------------------------------
        # Frame embedding is a form of PE

        # GCN ------------------------------------------------------------------
        if self.in_pos_vel_fusion == 1 or self.in_par_mot_fusion == 1:
            _in_ch = self.c1
        elif self.sem_position > 0 or self.sem_part > 0:
            _in_ch = self.c2
        else:
            _in_ch = self.c1
        if self.g_shared:
            self.gcn_g = GCNSpatialG(_in_ch,
                                     self.g_proj_dim,
                                     bias=bias,
                                     g_proj_shared=g_proj_shared)
        else:
            self.gcn_g1 = GCNSpatialG(_in_ch,
                                      self.g_proj_dim[0],
                                      bias=bias,
                                      g_proj_shared=g_proj_shared)
            self.gcn_g2 = GCNSpatialG(self.c2,
                                      self.g_proj_dim[1],
                                      bias=bias,
                                      g_proj_shared=g_proj_shared)
            self.gcn_g3 = GCNSpatialG(self.c3,
                                      self.g_proj_dim[2],
                                      bias=bias,
                                      g_proj_shared=g_proj_shared)

        self.gcn1 = GCNSpatial(_in_ch,
                               self.c2,
                               bias=bias,
                               kernel_size=gcn_t_kernel,
                               padding=gcn_t_kernel//2,
                               norm_mod=self.norm_mod)
        self.gcn2 = GCNSpatial(self.c2,
                               self.c3,
                               bias=bias,
                               kernel_size=gcn_t_kernel,
                               padding=gcn_t_kernel//2,
                               norm_mod=self.norm_mod)
        self.gcn3 = GCNSpatial(self.c3,
                               self.c3,
                               bias=bias,
                               kernel_size=gcn_t_kernel,
                               padding=gcn_t_kernel//2,
                               norm_mod=self.norm_mod)

        # ASPP -----------------------------------------------------------------
        if aspp is None or len(aspp) == 0:
            self.aspp = lambda x: x
        else:
            self.aspp = ASPP(self.c3,
                             self.c3,
                             bias=bias,
                             dilations=aspp,
                             norm_mod=self.norm_mod)

        # Frame level module ---------------------------------------------------
        self.smp = nn.AdaptiveMaxPool2d((1, num_segment))
        self.cnn = MLPTemporal(self.c3,
                               self.c4,
                               bias=bias,
                               t_kernel=t_kernel,
                               t_max_pool=t_max_pool,
                               norm_mod=self.norm_mod)
        self.tmp = nn.AdaptiveMaxPool2d((1, 1))
        self.do = nn.Dropout(dropout)

        self.fc = nn.Linear(self.c4, num_class)

        self.init()

        if self.in_part > 0 or self.in_motion > 0:
            self.register_buffer(
                'parts_3points_vec',
                torch.tensor(self.parts_3points, dtype=torch.int).reshape(-1)
            )

    def init_dr(self, *args, **kwargs):
        norm = kwargs.get('norm', None)
        norm = norm if norm is not None else self.norm_mod_1d
        return self.init_emb(*args, norm=norm, **kwargs)

    def init_emb(self,
                 mode: int,
                 num_point: int,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 bias: Optional[bool] = None,
                 norm: Optional[int] = None,
                 norm_mod: Optional[bool] = None):
        inter_channels = self.get_inter_channels(mode, self.c1)
        out_channels = out_channels if out_channels is not None else self.c1
        bias = bias if bias is not None else self.bias
        norm_mod = norm_mod if norm_mod is not None else self.norm_mod
        return Embedding(in_channels,
                         out_channels,
                         inter_channels=inter_channels,
                         bias=bias,
                         norm=norm,
                         norm_mod=norm_mod,
                         num_point=num_point,
                         mode=mode)

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        nn.init.constant_(self.gcn1.w1.block.conv.conv.weight, 0)
        nn.init.constant_(self.gcn2.w1.block.conv.conv.weight, 0)
        nn.init.constant_(self.gcn3.w1.block.conv.conv.weight, 0)

    def get_inter_channels(self, mode: int, ch: int) -> Union[list, int]:
        if mode == 3:
            return [ch, ch, ch]
        else:
            return ch

    def pad_zeros(self, x: Tensor) -> Tensor:
        return torch.cat([x.new(*x.shape[:-1], 1).zero_(), x], dim=-1)

    def forward(self,
                x: Tensor,
                s: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        """Model forward pass.

        Args:
            x (Tensor): 3D joint in_position of a sequence of skeletons.
            s (Tensor): Subject id for each frame.

        Returns:
            y (Tensor): logits.
            g (Tensor): Attention matrix for GCN.
        """
        bs, step, dim = x.shape
        assert dim % 3 == 0, "Only support input of xyz coordinates only."

        # Dynamic Representation -----------------------------------------------
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

        assert dy1 is not None or dy2 is not None

        # Joint and frame embeddings -------------------------------------------
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

        # Joint-level Module ---------------------------------------------------
        if dy1 is not None:
            if self.sem_position > 0:
                if self.in_pos_vel_fusion == 1:
                    x0 = dy1 + spa1  # n,c,v,t
                else:
                    x0 = torch.cat([dy1, spa1], 1)  # n,c,v,t
            else:
                x0 = dy1  # n,c,v,t

        if dy2 is not None:
            if self.sem_part > 0:
                if self.in_par_mot_fusion == 1:
                    x1 = dy2 + gro1  # n,c,v',t
                else:
                    x1 = torch.cat([dy2, gro1], 1)  # n,c,v',t
            else:
                x1 = dy2  # n,c,v',t

        if dy1 is not None and dy2 is not None:
            x = torch.cat([x0, x1], 2)  # n,c,v'+v,t
        elif dy1 is not None:
            x = x0
        elif dy2 is not None:
            x = x1
        else:
            raise ValueError("Unsupported input combination")

        if self.g_shared:
            g = self.gcn_g(x)
            x = self.gcn1(x, g)
            x = self.gcn2(x, g)
            x = self.gcn3(x, g)
        else:
            g1 = self.gcn_g1(x)
            x = self.gcn1(x, g1)
            g2 = self.gcn_g2(x)
            x = self.gcn2(x, g2)
            g3 = self.gcn_g3(x)
            x = self.gcn3(x, g3)
            g = [g1, g2, g3]

        # Frame-level Module ---------------------------------------------------
        if self.sem_frame > 0:
            x = x + tem1
        if self.subject > 0:
            x = x + sub1
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
                 norm_mod: Type[PyTorchModule] = nn.BatchNorm1d):
        super(DataNorm, self).__init__()
        self.bn = norm_mod(dim)  # channel dim * num_point

    def forward(self, x: Tensor) -> Tensor:
        bs, _, num_point, step = x.shape
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_point, step).contiguous()
        return x


class Embedding(Module):
    def __init__(self,
                 *args,
                 inter_channels: Union[list, int] = 0,
                 norm: Optional[Type[PyTorchModule]] = None,
                 norm_mod: Type[PyTorchModule] = nn.BatchNorm2d,
                 num_point: int = 25,
                 mode: int = 1,
                 ** kwargs):
        super(Embedding, self).__init__(*args, **kwargs)
        assert mode in [1, 2, 3]
        self.mode = mode

        if norm is not None:
            self.norm = DataNorm(self.in_channels * num_point, norm)
        else:
            self.norm = lambda x: x

        if mode == 1:
            self.cnn1 = Conv(self.in_channels,
                             inter_channels,
                             bias=self.bias,
                             activation=nn.ReLU)
            self.cnn2 = Conv(inter_channels,
                             self.out_channels,
                             bias=self.bias,
                             activation=nn.ReLU)
        elif mode == 2:
            # bert style
            self.cnn1 = Conv(self.in_channels,
                             self.out_channels,
                             bias=self.bias,
                             normalization=lambda: norm_mod(self.out_channels),
                             dropout=lambda: nn.Dropout2d(0.2))
        elif mode == 3:
            assert isinstance(inter_channels, list)
            inter_channels = \
                [self.in_channels] + inter_channels + [self.out_channels]
            for i in range(len(inter_channels)-1):
                setattr(self, f'cnn{i+1}', Conv(inter_channels[i],
                                                inter_channels[i+1],
                                                bias=self.bias,
                                                activation=nn.ReLU))

    def forward(self, x: Tensor) -> Tensor:
        # x: n,c,v,t
        x = self.norm(x)
        x = self.cnn1(x)
        if self.mode == 1:
            x = self.cnn2(x)
        elif self.mode == 3:
            x = self.cnn2(x)
            x = self.cnn3(x)
            x = self.cnn4(x)
        return x


class EmbeddingSubject(Module):
    def __init__(self,
                 *args,
                 inter_channels: Union[list, int] = 0,
                 num_subjects: int = 2,
                 mode: int = 1,
                 norm_mod: Type[PyTorchModule] = nn.BatchNorm2d,
                 **kwargs):
        super(EmbeddingSubject, self).__init__(*args, **kwargs)
        assert mode in [1, 2, 3, 4]
        self.mode = mode
        if mode == 1:
            # 2x MLP
            self.in_dim = self.in_channels
            self.cnn1 = Conv(self.in_channels,
                             inter_channels,
                             bias=self.bias,
                             activation=nn.ReLU)
            self.cnn2 = Conv(inter_channels,
                             self.out_channels,
                             bias=self.bias,
                             activation=nn.ReLU)
        elif mode == 2:
            # bert style
            self.in_dim = self.out_channels
            embedding = torch.empty(num_subjects, self.out_channels)
            nn.init.normal_(embedding, std=0.02)  # bert
            self.embedding = nn.Parameter(embedding)
            self.norm = norm_mod(self.out_channels)
            self.dropout = nn.Dropout2d(0.2)
        elif mode == 3:
            # 4x MLP
            self.in_dim = self.in_channels
            assert isinstance(inter_channels, list)
            inter_channels = \
                [self.in_channels] + inter_channels + [self.out_channels]
            for i in range(len(inter_channels)-1):
                setattr(self, f'cnn{i+1}', Conv(inter_channels[i],
                                                inter_channels[i+1],
                                                bias=self.bias,
                                                activation=nn.ReLU))
        elif mode == 4:
            # bert style and a conv
            self.in_dim = self.in_channels
            embedding = torch.empty(num_subjects, self.in_channels)
            nn.init.normal_(embedding, std=0.02)  # bert
            self.embedding = nn.Parameter(embedding)
            self.cnn1 = Conv(self.in_channels,
                             self.out_channels,
                             bias=self.bias,
                             activation=nn.ReLU)

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
            x = self.dropout(x)
        elif self.mode == 3:
            x = self.cnn1(x)
            x = self.cnn2(x)
            x = self.cnn3(x)
            x = self.cnn4(x)
        elif self.mode == 4:
            x = self.cnn1(x)
        return x


class MLPTemporal(Module):
    def __init__(self,
                 *args,
                 t_kernel: int = 3,
                 t_max_pool: int = 0,
                 norm_mod: Type[PyTorchModule] = nn.BatchNorm2d,
                 **kwargs):
        super(MLPTemporal, self).__init__(*args, **kwargs)
        self.t_max_pool = t_max_pool
        if t_max_pool:
            # self.maxp = nn.MaxPool2d(kernel_size=(1, t_kernel),
            #                          padding=(0, t_kernel//2))
            self.maxp = nn.MaxPool2d(kernel_size=(1, t_kernel),
                                     stride=(1, t_max_pool))
        else:

            self.cnn1 = Conv(self.in_channels,
                             self.in_channels,
                             kernel_size=t_kernel,
                             padding=t_kernel//2,
                             bias=self.bias,
                             activation=nn.ReLU,
                             normalization=lambda: norm_mod(self.in_channels),
                             dropout=lambda: nn.Dropout2d(0.2))
        self.cnn2 = Conv(self.in_channels,
                         self.out_channels,
                         bias=self.bias,
                         activation=nn.ReLU,
                         normalization=lambda: norm_mod(self.out_channels))

    def forward(self, x: Tensor) -> Tensor:
        # x: n,c,v,t ; v=1 due to SMP
        if self.t_max_pool:
            x = self.maxp(x)
        else:
            x = self.cnn1(x)
        x = self.cnn2(x)
        return x


class GCNSpatial(Module):
    def __init__(self,
                 *args,
                 norm_mod: Type[PyTorchModule] = nn.BatchNorm2d,
                 **kwargs):
        super(GCNSpatial, self).__init__(*args, **kwargs)
        self.bn = norm_mod(self.out_channels)
        self.relu = nn.ReLU()
        self.w1 = Conv(self.in_channels, self.out_channels, bias=self.bias)
        self.w2 = Conv(self.in_channels, self.out_channels, bias=self.bias,
                       kernel_size=self.kernel_size, padding=self.padding)

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        x1 = x.permute(0, 3, 2, 1).contiguous()  # n,t,v,c
        x1 = g.matmul(x1)
        x1 = x1.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
        x1 = self.w1(x1) + self.w2(x)  # z + residual
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        return x1


class GCNSpatialG(Module):
    def __init__(self,
                 *args,
                 g_proj_shared: bool = False,
                 **kwargs):
        super(GCNSpatialG, self).__init__(*args, **kwargs)
        self.gcn_g = Conv(self.in_channels, self.out_channels, bias=self.bias)
        if g_proj_shared:
            self.g2 = self.gcn_g
        else:
            self.g2 = Conv(self.in_channels, self.out_channels, bias=self.bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        gcn_g = self.gcn_g(x).permute(0, 3, 2, 1).contiguous()  # n,t,v,c
        g2 = self.g2(x).permute(0, 3, 1, 2).contiguous()  # n,t,c,v
        g3 = gcn_g.matmul(g2)  # n,t,v,v
        g4 = self.softmax(g3)
        return g4


class ASPP(Module):
    def __init__(self,
                 *args,
                 dilations: list = [1, 3, 5, 7],
                 norm_mod: Type[PyTorchModule] = nn.BatchNorm2d,
                 **kwargs):
        super(ASPP, self).__init__(*args, **kwargs)
        self.aspp = torch.nn.ModuleDict()
        self.pool = 0 in dilations
        if self.pool:
            self.aspp.update({
                'aspp_pool': Pool(self.in_channels,
                                  self.out_channels,
                                  bias=self.bias,
                                  pooling=nn.AdaptiveAvgPool2d(1),
                                  activation=nn.ReLU)
            })
        for dil in dilations:
            if dil == 0:
                continue
            self.aspp.update({
                f'aspp_{dil}': Conv(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=3,
                    padding=dil,
                    dilation=dil,
                    bias=self.bias,
                    activation=nn.ReLU,
                    normalization=lambda: norm_mod(self.out_channels),
                    deterministic=False
                )
            })
        self.proj = Conv(self.out_channels * len(dilations),
                         self.out_channels,
                         bias=self.bias,
                         normalization=lambda: norm_mod(self.out_channels),
                         dropout=lambda: nn.Dropout2d(0.2))

    def forward(self, x: Tensor) -> Tensor:
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
        x = self.proj(res)
        return x


if __name__ == '__main__':

    batch_size = 64

    from model.resource.tests import test
    test(SGN)

    exit(1)

    model = SGN(num_segment=20, in_part=True, in_motion=True,
                subject=True, aspp=[0, 1, 5, 9])
    print(model)

    model = SGN(num_segment=20, in_part=True, in_motion=True,
                subject=True, aspp=[0, 1, 5, 9]).cuda()
    inputs = torch.ones(batch_size, 20, 75).cuda()
    subjects = torch.ones(batch_size, 20, 1).cuda()
    # subjects = None
    model(inputs, subjects)
    # with torch.autograd.profiler.profile() as prof:
    #     with torch.autograd.profiler.record_function("model_inference"):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        # with record_function("model_inference"):
        model(inputs, subjects)
    # print(prof.key_averages(group_by_input_shape=True).table(
    #     sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
