# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Taken from :
# https://github.com/microsoft/SGN/blob/master/model.py

# Code refractored

# CODE FREEZED ON 01.04.2022

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import profile, record_function, ProfilerActivity

import math
import time
from collections import OrderedDict
from tqdm import tqdm
from typing import Tuple, Optional, Union

from model.resource.common_ntu import *
from utils.utils import *


class Module(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False):
        super(Module, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.bias = bias


class SGN(nn.Module):

    c1, c2, c3, c4 = c1, c2, c3, c4

    # NTU
    parts_3points_wholebody = parts_3points_wholebody

    def __init__(self,
                 num_class: int = 60,
                 num_point: int = 25,
                 in_channels: int = 3,
                 seg: int = 20,
                 bias: bool = True,

                 c_multiplier: int = 1,
                 dropout: float = 0.0,

                 part: Union[bool, int] = 0,
                 motion: Union[bool, int] = 0,
                 subject: bool = False,

                 g_proj_shared: bool = False,

                 t_kernel: int = 3,
                 t_max_pool: bool = False,
                 aspp: list = None,
                 ):
        super(SGN, self).__init__()

        self.c1 *= c_multiplier  # pos,vel,joint embed
        self.c2 *= c_multiplier  # G,gcn
        self.c3 *= c_multiplier  # gcn
        self.c4 *= c_multiplier  # gcn

        self.num_class = num_class
        self.num_point = num_point
        self.in_channels = in_channels
        self.seg = seg
        self.bias = bias

        self.part = bool2int(part)
        self.motion = bool2int(motion)
        self.subject = subject

        self.g_proj_shared = g_proj_shared
        self.t_kernel = t_kernel
        self.t_max_pool = t_max_pool

        self.pos_embed = embed(in_channels,
                               self.c1,
                               inter_channels=self.c1,
                               num_point=num_point,
                               norm=True,
                               bias=bias)
        self.vel_embed = embed(in_channels,
                               self.c1,
                               inter_channels=self.c1,
                               num_point=num_point,
                               norm=True,
                               bias=bias)

        if self.part == 1:
            self.par_embed = embed(in_channels*3,
                                   self.c1,
                                   inter_channels=self.c1,
                                   num_point=len(self.parts_3points),
                                   norm=True,
                                   bias=bias)
            if self.motion == 1:
                self.mot_embed = embed(in_channels,
                                       self.c1,
                                       inter_channels=self.c1,
                                       num_point=len(self.parts_3points),
                                       norm=True,
                                       bias=bias)
            elif self.motion == 2:
                self.mot_embed = embed(in_channels*3,
                                       self.c1,
                                       inter_channels=self.c1,
                                       num_point=len(self.parts_3points),
                                       norm=True,
                                       bias=bias)

        if self.subject:
            self.sub_embed = embed_subject(1,
                                           self.c3,
                                           inter_channels=self.c1,
                                           num_subjects=2,
                                           bias=bias)

        self.spa = one_hot(num_point, seg, mode=0)
        self.tem = one_hot(seg, num_point, mode=1)
        self.spa_embed = embed(num_point,
                               self.c1,
                               inter_channels=self.c1,
                               num_point=num_point,
                               norm=False,
                               bias=bias)
        self.tem_embed = embed(seg,
                               self.c3,
                               inter_channels=self.c1,
                               num_point=num_point,
                               norm=False,
                               bias=bias)

        if self.part == 1:
            self.tem = one_hot(seg, num_point+len(self.parts_3points), mode=1)
            self.gro = one_hot(len(self.parts_3points), seg, mode=0)
            self.gro_embed = embed(len(self.parts_3points),
                                   self.c1,
                                   inter_channels=self.c1,
                                   num_point=len(self.parts_3points),
                                   norm=False,
                                   bias=bias)

        self.compute_g1 = compute_g_spa(self.c2,
                                        self.c3,
                                        bias=bias,
                                        g_proj_shared=g_proj_shared)
        self.gcn1 = gcn_spa(self.c2, self.c2, bias=bias)
        self.gcn2 = gcn_spa(self.c2, self.c3, bias=bias)
        self.gcn3 = gcn_spa(self.c3, self.c3, bias=bias)

        if aspp is None or len(aspp) == 0:
            self.aspp = lambda x: x
        else:
            self.aspp = atrous_spatial_pyramid_pooling(
                self.c3,
                self.c3,
                bias=bias,
                dilations=aspp)

        self.smp = nn.AdaptiveMaxPool2d((1, seg))
        self.cnn = local(self.c3, self.c4, bias=bias,
                         t_kernel=t_kernel, t_max_pool=t_max_pool)
        self.tmp = nn.AdaptiveMaxPool2d((1, 1))
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear(self.c4, num_class)

        self.init()

        if self.part == 1:
            self.register_buffer('parts_3points_vec',
                                 torch.tensor(self.parts_3points,
                                              dtype=torch.int).reshape(-1))

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w1.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w1.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w1.cnn.weight, 0)

    def pad_zeros(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.new(*x.shape[:-1], 1).zero_(), x], dim=-1)

    def forward(self,
                x: torch.Tensor,
                s: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model forward pass.

        Args:
            x (torch.Tensor): 3D joint position of a sequence of skeletons.
            s (torch.Tensor): Subject id for each frame.

        Returns:
            y (torch.Tensor): logits.
            g (torch.Tensor): Attention matrix for GCN.
        """
        bs, step, dim = x.shape
        assert dim % 3 == 0, "Only support input of xyz coordinates only."

        # Dynamic Representation
        num_point = dim // 3
        x1 = x.view((bs, step, num_point, 3))  # n,t,v,c
        x = x1.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
        dif = x[:, :, :, 1:] - x[:, :, :, 0:-1]  # n,c,v,t-1
        dif = self.pad_zeros(dif)
        pos = self.pos_embed(x)
        dif = self.vel_embed(dif)
        dy1 = pos + dif  # n,c,v,t

        if self.part == 1:
            par = torch.index_select(x1, 2, self.parts_3points_vec)  # n,t,v+,c
            par = par.view((bs, step, -1, 3, self.in_channels))  # n,t,v+,3,c
            mid = par.mean(dim=-2, keepdim=True)  # n,t,v+,1,c
            par1 = par - mid  # n,t,v+,3,c
            par = par1.view((bs, step, -1, self.in_channels*3))  # n,t,v+,c+
            par = par.permute(0, 3, 2, 1).contiguous()  # n,c+,v+,t
            dy2 = self.par_embed(par)  # n,c,v+,t

            if self.motion == 1:
                mid = mid.squeeze(-2)  # n,t,v+,c
                mid = mid.permute(0, 3, 2, 1).contiguous()  # n,c,v+,t
                mot = mid[:, :, :, 1:] - mid[:, :, :, 0:-1]  # n,c,v+,t-1
                mot = self.pad_zeros(mot)
                mot = self.mot_embed(mot)
                dy2 += mot
            elif self.motion == 2:
                # mid = mid  # n,t,v+,1,c
                # par1 = par1  # n,t,v+,3,c
                mot = par1[:, 1:] - mid[:, :-1]  # n,t-1,v+,3,c
                mot = mot.view((*mot.shape[:3], -1))  # n,t-1,v+,c+
                mot = mot.permute(0, 3, 2, 1).contiguous()  # n,c,v+,t-1
                mot = self.pad_zeros(mot)  # n,c,v+,t
                mot = self.mot_embed(mot)
                dy2 += mot

        # Joint and frame embeddings
        tem1 = self.tem_embed(self.tem(bs))
        spa1 = self.spa_embed(self.spa(bs))
        if self.part == 1:
            gro1 = self.gro_embed(self.gro(bs))
        if self.subject:
            s = s.view((bs, step, 1, 1))  # n,t,v,c
            s = s.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
            sub1 = self.sub_embed(s)

        # Joint-level Module
        x = torch.cat([dy1, spa1], 1)  # n,c,v,t
        if self.part == 1:
            x1 = torch.cat([dy2, gro1], 1)  # n,c,v',t
            x = torch.cat([x, x1], 2)  # n,c,v'+v,t
        g = self.compute_g1(x)
        x = self.gcn1(x, g)
        x = self.gcn2(x, g)
        x = self.gcn3(x, g)

        # Frame-level Module
        x = x + tem1
        x = self.smp(x)
        if self.subject:
            x = x + sub1
        x = self.aspp(x)
        x = self.cnn(x)

        # Classification
        y = self.tmp(x)
        y = torch.flatten(y, 1)
        y = self.do(y)
        y = self.fc(y)

        return y, g


class one_hot(nn.Module):
    def __init__(self, dim_eye: int, dim_length: int, mode: int):
        super(one_hot, self).__init__()
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

    def forward(self, bs: int) -> torch.Tensor:
        x = self.onehot.repeat(bs, 1, 1, 1)
        return x


class norm_data(nn.Module):
    def __init__(self, dim: int):
        super(norm_data, self).__init__()
        self.bn = nn.BatchNorm1d(dim)  # channel dim * num_point

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, _, num_point, step = x.shape
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_point, step).contiguous()
        return x


class embed_subject(Module):
    def __init__(self,
                 *args,
                 num_subjects=2,
                 inter_channels: int = 0,
                 **kwargs):
        super(embed_subject, self).__init__(*args, **kwargs)
        embedding = torch.empty(num_subjects, inter_channels)
        nn.init.normal_(embedding, std=0.02)  # bert
        self.embedding = nn.Parameter(embedding)
        self.cnn1 = cnn1x1(inter_channels, self.out_channels, bias=self.bias)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, _, _, step = x.shape  # n,c,v,t
        x1 = x.reshape(-1).unsqueeze(-1)  # nt,1
        x1 = x1.expand(x1.shape[0], self.embedding.shape[-1]).type(torch.int64)
        x1 = torch.gather(self.embedding, 0, x1)
        x1 = x1.reshape((bs, step, 1, self.embedding.shape[-1]))
        x1 = x1.transpose(1, -1)
        x = self.cnn1(x1)
        x = self.relu(x)
        return x


class embed(Module):
    def __init__(self,
                 *args,
                 norm: bool = False,
                 inter_channels: int = 0,
                 num_point: int = 25,
                 **kwargs):
        super(embed, self).__init__(*args, **kwargs)
        if norm:
            self.norm = norm_data(self.in_channels * num_point)
        else:
            self.norm = lambda x: x
        self.cnn1 = cnn1x1(self.in_channels, inter_channels, bias=self.bias)
        self.relu = nn.ReLU()
        self.cnn2 = cnn1x1(inter_channels, self.out_channels, bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: n,c,v,t
        x = self.norm(x)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.relu(x)
        return x


class cnn1xn(Module):
    def __init__(self, *args, deterministic=True, **kwargs):
        super(cnn1xn, self).__init__(*args, **kwargs)
        assert isinstance(self.kernel_size, int)
        assert isinstance(self.padding, int)
        assert isinstance(self.dilation, int)
        self.cnn = nn.Conv2d(self.in_channels,
                             self.out_channels,
                             kernel_size=(1, self.kernel_size),
                             padding=(0, self.padding),
                             dilation=self.dilation,
                             bias=self.bias)
        self.deterministic = deterministic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.deterministic:
            torch.backends.cudnn.deterministic = False
            # torch.backends.cudnn.benchmark = True
        x = self.cnn(x)
        if not self.deterministic:
            torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = True
        return x


class cnn1x1(cnn1xn):
    def __init__(self, *args, **kwargs):
        kwargs['kernel_size'] = 1
        kwargs['padding'] = 0
        kwargs['dilation'] = 1
        super(cnn1x1, self).__init__(*args, **kwargs)


class local(Module):
    def __init__(self,
                 *args,
                 t_kernel: int = 3,
                 t_max_pool: bool = False,
                 **kwargs):
        super(local, self).__init__(*args, **kwargs)
        self.t_max_pool = t_max_pool
        if t_max_pool:
            self.maxp = nn.MaxPool2d(kernel_size=(1, t_kernel),
                                     padding=(0, t_kernel//2))
        else:
            self.cnn1 = cnn1xn(self.in_channels,
                               self.in_channels,
                               kernel_size=t_kernel,
                               padding=t_kernel//2,
                               bias=self.bias)
            self.bn1 = nn.BatchNorm2d(self.in_channels)
            self.dropout = nn.Dropout2d(0.2)
        self.cnn2 = cnn1x1(self.in_channels,
                           self.out_channels,
                           bias=self.bias)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: n,c,v,t ; v=1 due to SMP
        if self.t_max_pool:
            x = self.maxp(x)
        else:
            x = self.cnn1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class gcn_spa(Module):
    def __init__(self, *args, **kwargs):
        super(gcn_spa, self).__init__(*args, **kwargs)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()
        self.w1 = cnn1x1(self.in_channels, self.out_channels, bias=self.bias)
        self.w2 = cnn1xn(self.in_channels, self.out_channels, bias=self.bias,
                         kernel_size=self.kernel_size, padding=self.padding)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x1 = x.permute(0, 3, 2, 1).contiguous()  # n,t,v,c
        x1 = g.matmul(x1)
        x1 = x1.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
        x1 = self.w1(x1) + self.w2(x)  # z + residual
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        return x1


class compute_g_spa(Module):
    def __init__(self, *args, g_proj_shared: bool = False, **kwargs):
        super(compute_g_spa, self).__init__(*args, **kwargs)
        self.g1 = cnn1x1(self.in_channels, self.out_channels, bias=self.bias)
        if g_proj_shared:
            self.g2 = self.g1
        else:
            self.g2 = cnn1x1(self.in_channels,
                             self.out_channels,
                             bias=self.bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g1 = self.g1(x).permute(0, 3, 2, 1).contiguous()  # n,t,v,c
        g2 = self.g2(x).permute(0, 3, 1, 2).contiguous()  # n,t,c,v
        g3 = g1.matmul(g2)  # n,t,v,v
        g4 = self.softmax(g3)
        return g4


class atrous_spatial_pyramid_pooling(Module):
    def __init__(self,
                 *args,
                 dilations: list = [1, 3, 5, 7],
                 **kwargs):
        super(atrous_spatial_pyramid_pooling, self).__init__(*args, **kwargs)
        self.aspp = torch.nn.ModuleDict()

        self.pool = 0 in dilations
        if self.pool:
            self.aspp.update({
                'aspp_pool':
                nn.Sequential(
                    OrderedDict([
                        (f'avg_pool', nn.AdaptiveAvgPool2d(1)),
                        (f'conv_pool', cnn1x1(self.in_channels,
                                              self.out_channels,
                                              bias=self.bias)),
                        (f'relu_pool', nn.ReLU()),
                    ])
                )
            })

        for dil in dilations:
            if dil == 0:
                continue
            self.aspp.update({
                f'aspp_{dil}':
                nn.Sequential(
                    OrderedDict([
                        (f'conv_{dil}', cnn1xn(self.in_channels,
                                               self.out_channels,
                                               kernel_size=3,
                                               padding=dil,
                                               dilation=dil,
                                               bias=self.bias,
                                               deterministic=False)),
                        (f'bn_{dil}', nn.BatchNorm2d(self.out_channels)),
                        (f'relu_{dil}', nn.ReLU()),
                    ])
                )
            })

        branches = len(dilations) + 1 if self.pool else len(dilations)
        self.proj = cnn1x1(self.out_channels * branches,
                           self.out_channels,
                           bias=self.bias)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.bn(x)
        x = self.dropout(x)
        return x


if __name__ == '__main__':

    batch_size = 64

    # model = SGN(seg=20, part=True, motion=True, aspp=[0, 1, 5, 9]).cuda()
    # for i in tqdm(range(100)):
    #     inputs = torch.ones(batch_size, 20, 75).cuda()
    #     model(inputs)

    model = SGN(seg=20, part=True, motion=True,
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
    # print(prof.key_averages().table())

    # model = SGN(seg=20, part=True, motion=True, aspp=[0, 1, 5, 9]).cuda()
    # print(model)
    # s = time.time()
    # for i in tqdm(range(100)):
    #     model(torch.ones(batch_size, 20, 75).cuda())
    # print("Time :", time.time()-s)

    # model = SGN(seg=20, part=True, motion=True, aspp=[]).cuda()
    # s = time.time()
    # for i in tqdm(range(100)):
    #     model(torch.ones(batch_size, 20, 75).cuda())
    # print("Time :", time.time()-s)

    # model = SGN(seg=20, part=True, motion=True, aspp=[0, 1, 5, 9]).cuda()
    # s = time.time()
    # for i in tqdm(range(100)):
    #     model(torch.ones(batch_size, 20, 75).cuda())
    # print("Time :", time.time()-s)

    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
