# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Taken from :
# https://github.com/microsoft/SGN/blob/master/model.py

# Code base on:
# An efficient self‑attention network for skeleton‑based action recognition

import torch
from torch import nn

from model.sgn_v2 import SGN as SGNBase
from model.sgn_v2 import embed
from model.sgn_v2 import gcn_spa


class SGN(SGNBase):

    def __init__(self,
                 gcn_t_kernel: int = 3,
                 **kwargs):

        super(SGN, self).__init__(**kwargs)

        self.tem_embed = embed(self.seg,
                               self.c2,
                               inter_channels=self.c1,
                               num_point=self.num_point,
                               norm=False,
                               bias=self.bias)
        self.gcn1 = gcn_spa(self.c2,
                            self.c2,
                            bias=self.bias,
                            kernel_size=gcn_t_kernel,
                            padding=gcn_t_kernel//2)
        self.gcn2 = gcn_spa(self.c2,
                            self.c3,
                            bias=self.bias,
                            kernel_size=gcn_t_kernel,
                            padding=gcn_t_kernel//2)
        self.gcn3 = gcn_spa(self.c3,
                            self.c4,
                            bias=self.bias,
                            kernel_size=gcn_t_kernel,
                            padding=gcn_t_kernel//2)
        self.fc = nn.Linear(self.c4, self.num_class)

        del self.cnn

        self.init()

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
        # Frame-level Module
        x = torch.cat([dy, spa1], 1)  # n,c,v,t
        x = x + tem1
        g = self.compute_g1(x)
        x = self.gcn1(x, g)
        x = self.gcn2(x, g)
        x = self.gcn3(x, g)

        # Classification
        x = self.smp(x)
        y = self.tmp(x)
        y = torch.flatten(y, 1)
        if hasattr(self, 'dropout'):
            y = self.dropout(y)
        y = self.fc(y)

        return y, g


if __name__ == '__main__':
    batch_size = 2
    model = SGN(num_class=60,
                num_point=25,
                in_channels=3,
                seg=20,
                bias=True,
                g_proj_shared=False,
                gcn_t_kernel=3,
                dropout=0.0,
                c_multiplier=2).cuda()
    model(torch.ones(batch_size, 20, 75).cuda())
