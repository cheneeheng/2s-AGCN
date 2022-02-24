import torch
from torch import nn
from torchinfo import summary

import numpy as np
from typing import Optional

from model.aagcn import import_class
from model.aagcn import TCNGCNUnit
from model.aagcn import BaseModel


# ------------------------------------------------------------------------------
# Blocks
# ------------------------------------------------------------------------------
class MHAUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_heads: int = 1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=0.,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=True
        )
        self.norm = nn.LayerNorm(
            normalized_shape=in_channels,
            eps=1e-05,
            elementwise_affine=True
        )

    def forward(self, x: torch.Tensor, original_shape: bool = True):
        # x : N C T V, N = N*M
        N, _, T, V = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # N T C V
        x = x.view(N, T, -1)  # N T CV
        mha_x, mha_attn = self.mha(x, x, x)
        x = x + mha_x
        x = self.norm(x)  # N T CV
        if original_shape:
            x = x.view(N, T, -1, V)  # N T C V
            x = x.permute(0, 2, 1, 3).contiguous()  # N C T V
        return x, mha_attn


# ------------------------------------------------------------------------------
# Network
# - using MHA at the end
# ------------------------------------------------------------------------------
class Model(BaseModel):
    def __init__(self,
                 num_class: int = 60,
                 num_point: int = 25,
                 num_person: int = 2,
                 num_subset: int = 3,
                 graph: Optional[str] = None,
                 graph_args: dict = dict(),
                 in_channels: int = 3,
                 drop_out: int = 0,
                 adaptive: bool = True,
                 attention: bool = True,
                 gbn_split: Optional[int] = None,
                 num_heads: int = 1,
                 postprocess_type: Optional[str] = 'GAP-TV',
                 model_layers: int = 10):
        super().__init__(num_class, num_point, num_person,
                         in_channels, drop_out, adaptive, gbn_split)

        assert postprocess_type in ['GAP-T', 'GAP-TV', 'Flat']
        self.postprocess_type = postprocess_type

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        def _TCNGCNUnit(_in, _out, stride=1, residual=True):
            return TCNGCNUnit(_in,
                              _out,
                              self.graph.A,
                              num_subset=num_subset,
                              stride=stride,
                              residual=residual,
                              adaptive=self.adaptive_fn,
                              attention=attention,
                              gbn_split=gbn_split)

        self.init_model_backbone(model_layers=model_layers,
                                 tcngcn_unit=_TCNGCNUnit)

        if self.postprocess_type == 'GAP-T':
            self.init_fc(256*num_point, num_class)
        elif self.postprocess_type == 'GAP-TV':
            self.init_fc(256, num_class)
        elif self.postprocess_type == 'Flat':
            self.init_flat_postprocess(256, num_point)
            self.init_fc(256*num_person, num_class)

        self.mha = MHAUnit(in_channels=256*num_point,
                           num_heads=num_heads)

    def init_flat_postprocess(self, in_channels, num_point):
        self.proj1 = nn.Linear(in_channels*num_point, in_channels//2)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.LayerNorm(
            normalized_shape=in_channels//2,
            eps=1e-05,
            elementwise_affine=True
        )
        self.proj2 = nn.Linear(in_channels//2*75, in_channels)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.LayerNorm(
            normalized_shape=in_channels,
            eps=1e-05,
            elementwise_affine=True
        )

    # def forward_mlp(self, x):
    #     x = self.proj1(x)
    #     x = self.relu1(x)
    #     x = self.drop_out1(x)
    #     x = self.proj2(x)

    def forward_postprocess(self, x, size):
        # x : NM C T V
        N, C, T, V, M = size
        c_new = x.size(1)
        if self.postprocess_type == 'GAP-T':
            x, a = self.mha(x, False)  # n,t,cv
            x = x.view(N, M, -1, c_new, V)  # n,m,t,c,v
            x = x.mean(2).mean(1)  # n,c,v
            x = x.view(N, -1)  # n,cv
        elif self.postprocess_type == 'GAP-TV':
            x, a = self.mha(x, True)  # n,c,t,v
            x = x.view(N, M, c_new, -1)  # n,m,c,tv
            x = x.mean(3).mean(1)  # n,c
        elif self.postprocess_type == 'Flat':
            x, a = self.mha(x, False)  # n,t,cv
            x = self.norm1(self.relu1(self.proj1(x)))  # n,t,c'
            x = x.view(N*M, -1)  # n,tc'
            x = self.norm2(self.relu2(self.proj2(x)))  # n,c
            x = x.view(N, -1)  # n,mc
        return x, a


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model(torch.ones((1, 3, 300, 25, 2)))
