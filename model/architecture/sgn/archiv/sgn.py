# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Taken from :
# https://github.com/microsoft/SGN/blob/master/model.py

# Original code with slight style/naming/argument refractoring

from torch import nn
import torch
import math


class SGN(nn.Module):
    def __init__(self,
                 num_class: int = 60,
                 num_point: int = 25,
                 in_channels: int = 3,
                 seg: int = 20,
                 bias: bool = True):
        super(SGN, self).__init__()

        self.c1 = 64
        self.c2 = 128
        self.c3 = 256
        self.seg = seg

        self.joint_embed = embed(in_channels,
                                 self.c1,
                                 norm_dim=in_channels*num_point,
                                 bias=bias)
        self.dif_embed = embed(in_channels,
                               self.c1,
                               norm_dim=in_channels*num_point,
                               bias=bias)

        # self.spa = one_hot(num_point, self.seg, 'spa')
        # self.tem = one_hot(self.seg, num_point, 'tem')

        self.spa = self.one_hot(
            1, num_point, self.seg).permute(0, 3, 2, 1).cuda()
        self.tem = self.one_hot(
            1, self.seg, num_point).permute(0, 3, 1, 2).cuda()

        self.tem_embed = embed(self.seg, self.c3, norm_dim=0, bias=bias)
        self.spa_embed = embed(num_point, self.c1, norm_dim=0, bias=bias)

        self.compute_g1 = compute_g_spa(self.c2, self.c3, bias=bias)
        self.gcn1 = gcn_spa(self.c2, self.c2, bias=bias)
        self.gcn2 = gcn_spa(self.c2, self.c3, bias=bias)
        self.gcn3 = gcn_spa(self.c3, self.c3, bias=bias)

        self.cnn = local(self.c3, self.c3 * 2, bias=bias, seg=seg)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(self.c3 * 2, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

    def forward(self, x):

        # Dynamic Representation
        bs, step, dim = x.size()
        num_joints = dim // 3
        x = x.view((bs, step, num_joints, 3))
        x = x.permute(0, 3, 2, 1).contiguous()
        dif = x[:, :, :, 1:] - x[:, :, :, 0:-1]
        dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(),
                         dif], dim=-1)
        pos = self.joint_embed(x)

        tem1 = self.tem_embed(self.tem.repeat(bs, 1, 1, 1))
        spa1 = self.spa_embed(self.spa.repeat(bs, 1, 1, 1))
        dif = self.dif_embed(dif)
        dy = pos + dif
        # Joint-level Module
        x = torch.cat([dy, spa1], 1)
        g = self.compute_g1(x)
        x = self.gcn1(x, g)
        x = self.gcn2(x, g)
        x = self.gcn3(x, g)
        # Frame-level Module
        x = x + tem1
        x = self.cnn(x)
        # Classification
        y = self.maxpool(x)
        y = torch.flatten(y, 1)
        y = self.fc(y)

        attn = g

        return y, attn

    def one_hot(self, bs, spa, tem):
        y_onehot = torch.eye(spa, spa)
        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)
        return y_onehot


class norm_data(nn.Module):
    def __init__(self, dim=64):
        super(norm_data, self).__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x


class embed(nn.Module):
    def __init__(self, dim=3, dim1=128, norm_dim=75, bias=False):
        super(embed, self).__init__()

        if norm_dim > 0:
            self.cnn = nn.Sequential(
                norm_data(norm_dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x


class cnn1x1(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x


class local(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=False, seg: int = 20):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, seg))
        self.cnn1 = nn.Conv2d(dim1, dim1,
                              kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)

    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x


class compute_g_spa(nn.Module):
    def __init__(self, dim1=64*3, dim2=64*3, bias=False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):
        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g


if __name__ == '__main__':
    batch_size = 2
    model = SGN(seg=100).cuda()
    model(torch.ones(batch_size, 100, 75).cuda())
