# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Taken from :
# https://github.com/microsoft/SGN/blob/master/model.py

# Code refractored

from torch import nn
import torch
import math


class Module(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 bias: bool = False):
        super(Module, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.kernel_size = kernel_size
        self.bias = bias


class SGN(nn.Module):
    def __init__(self,
                 num_class: int = 60,
                 num_point: int = 25,
                 in_channels: int = 3,
                 seg: int = 20,
                 bias: bool = True,
                 g_proj_shared: bool = False):
        super(SGN, self).__init__()

        self.num_class = num_class
        self.num_point = num_point
        self.in_channels = in_channels
        self.seg = seg
        self.bias = bias
        self.g_proj_shared = g_proj_shared

        self.c1 = 64
        self.c2 = 128
        self.c3 = 256
        self.c4 = 512

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

        self.spa = one_hot(num_point, seg, mode=0)
        self.tem = one_hot(seg, num_point, mode=1)

        self.tem_embed = embed(seg,
                               self.c3,
                               inter_channels=self.c1,
                               num_point=num_point,
                               norm=False,
                               bias=bias)
        self.spa_embed = embed(num_point,
                               self.c1,
                               inter_channels=self.c1,
                               num_point=num_point,
                               norm=False,
                               bias=bias)

        self.compute_g1 = compute_g_spa(self.c2, self.c3, bias=bias,
                                        g_proj_shared=g_proj_shared)
        self.gcn1 = gcn_spa(self.c2, self.c2, bias=bias)
        self.gcn2 = gcn_spa(self.c2, self.c3, bias=bias)
        self.gcn3 = gcn_spa(self.c3, self.c3, bias=bias)

        self.smp = nn.AdaptiveMaxPool2d((1, seg))
        self.cnn = local(self.c3, self.c4, bias=bias)
        self.tmp = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(self.c4, num_class)

        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w1.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w1.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w1.cnn.weight, 0)

    def forward(self, x: torch.Tensor):
        bs, step, dim = x.shape
        assert dim % 3 == 0, "Only support input of xyz coordinates only."

        # Dynamic Representation
        num_point = dim // 3
        x = x.view((bs, step, num_point, 3))  # n,t,v,c
        x = x.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
        dif = x[:, :, :, 1:] - x[:, :, :, 0:-1]  # n,c,v,t-1
        dif = torch.cat([dif.new(*dif.shape[:-1], 1).zero_(), dif], dim=-1)
        pos = self.pos_embed(x)
        dif = self.vel_embed(dif)
        dy = pos + dif  # n,c,v,t

        # Joint and frame embeddings
        tem1 = self.tem_embed(self.tem(bs))
        spa1 = self.spa_embed(self.spa(bs))

        # Joint-level Module
        x = torch.cat([dy, spa1], 1)  # n,c,v,t
        g = self.compute_g1(x)
        x = self.gcn1(x, g)
        x = self.gcn2(x, g)
        x = self.gcn3(x, g)

        # Frame-level Module
        x = x + tem1
        x = self.smp(x)
        x = self.cnn(x)

        # Classification
        y = self.tmp(x)
        y = torch.flatten(y, 1)
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
        x = self.norm(x)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.relu(x)
        return x


class cnn1xn(Module):
    def __init__(self, *args, **kwargs):
        super(cnn1xn, self).__init__(*args, **kwargs)
        assert isinstance(self.kernel_size, int)
        assert isinstance(self.padding, int)
        self.cnn = nn.Conv2d(self.in_channels,
                             self.out_channels,
                             kernel_size=(1, self.kernel_size),
                             padding=(0, self.padding),
                             bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        return x


class cnn1x1(cnn1xn):
    def __init__(self, *args, **kwargs):
        kwargs['kernel_size'] = 1
        kwargs['padding'] = 0
        super(cnn1x1, self).__init__(*args, **kwargs)


class local(Module):
    def __init__(self, *args, **kwargs):
        super(local, self).__init__(*args, **kwargs)
        self.cnn1 = cnn1xn(self.in_channels,
                           self.in_channels,
                           kernel_size=3,
                           padding=1,
                           bias=self.bias)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.cnn2 = cnn1x1(self.in_channels,
                           self.out_channels,
                           bias=self.bias)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


if __name__ == '__main__':
    batch_size = 2
    model = SGN(seg=100)
    model(torch.ones(batch_size, 100, 75))
