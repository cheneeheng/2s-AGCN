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
                 out_channels: int = 0,
                 num_heads: int = 1,
                 gbn_split: Optional[int] = None):
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
# - creates multiple attentions instead of one in AdaptiveGCN
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
                 postprocess_type: Optional[str] = 'GAP-TV'):
        super().__init__(num_class, num_point, num_person,
                         in_channels, drop_out, adaptive, gbn_split)

        assert postprocess_type in ['GAP-T', 'GAP-TV']
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

        self.l1 = _TCNGCNUnit(3, 64, residual=False)
        self.l2 = _TCNGCNUnit(64, 64)
        self.l3 = _TCNGCNUnit(64, 64)
        self.l4 = _TCNGCNUnit(64, 64)
        self.l5 = _TCNGCNUnit(64, 128, stride=2)
        self.l6 = _TCNGCNUnit(128, 128)
        self.l7 = _TCNGCNUnit(128, 128)
        self.l8 = _TCNGCNUnit(128, 256, stride=2)
        self.l9 = _TCNGCNUnit(256, 256)
        self.l10 = _TCNGCNUnit(256, 256)

        if self.postprocess_type == 'GAP-T':
            self.init_fc(256*num_point, num_class)
        elif self.postprocess_type == 'GAP-TV':
            self.init_fc(256, num_class)

        self.mha = MHAUnit(in_channels=256*num_point,
                           num_heads=num_heads)

    def forward_postprocess(self, x, size):
        # x : NM C T V
        N, C, T, V, M = size
        c_new = x.size(1)
        if self.postprocess_type == 'GAP-T':
            x, a = self.mha(x, False)  # N T CV
            x = x.view(N, M, -1, c_new, V)  # n,m,t,c,v
            x = x.mean(2).mean(1)  # n,c,v
            x = x.view(N, -1)  # n,cv
        elif self.postprocess_type == 'GAP-TV':
            x, a = self.mha(x, True)  # N C T V
            x = x.view(N, M, c_new, -1)  # n,m,c,tv
            x = x.mean(3).mean(1)  # n,c
        return x


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model(torch.ones((1, 3, 300, 25, 2)))
