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
# - pre-LN : On Layer Normalization in the Transformer Architecture
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

    def forward(self, x: torch.Tensor):
        # x : N, L, C
        x = self.norm(x)
        mha_x, mha_attn = self.mha(x, x, x)
        x = x + mha_x
        return x, mha_attn


class FFNUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 inter_channels: int,
                 out_channels: int,
                 skip: bool = True):
        super().__init__()
        self.skip = skip
        self.l1 = nn.Linear(in_channels, inter_channels)
        self.l2 = nn.Linear(inter_channels, out_channels)
        self.re = nn.ReLU()
        self.n1 = nn.LayerNorm(
            normalized_shape=in_channels,
            eps=1e-05,
            elementwise_affine=True
        )

    def forward(self, x: torch.Tensor):
        # x : N, L, C
        if self.skip:
            x = x + self.l2(self.re(self.l1(self.n1(x))))
        else:
            x = self.l2(self.re(self.l1(self.n1(x))))
        return x


class TransformerUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_heads: int = 1):
        super().__init__()
        self.ffn1 = FFNUnit(in_channels=in_channels,
                            inter_channels=(in_channels+out_channels)//8,
                            out_channels=out_channels,
                            skip=False)
        self.mha1 = MHAUnit(in_channels=out_channels, num_heads=num_heads)
        self.ffn2 = FFNUnit(in_channels=out_channels,
                            inter_channels=out_channels * 4,
                            out_channels=out_channels,
                            skip=True)

    def forward(self, x: torch.Tensor):
        # x : N, L, C
        ffn1_x = self.ffn1(x)  # projections
        mha1_x, mha1_attn = self.mha1(ffn1_x)
        ffn2_x = self.ffn2(ffn1_x + mha1_x)
        return ffn2_x, mha1_attn


# ------------------------------------------------------------------------------
# Network
# - uses transformer with fetaure projection.
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
                 model_layers: int = 10):
        super().__init__(num_class, num_point, num_person,
                         in_channels, drop_out, adaptive, gbn_split)

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
        self.transformer1 = TransformerUnit(
            in_channels=256*num_point*num_person,
            out_channels=256,
            num_heads=num_heads
        )
        self.init_fc(256, num_class)

    def forward_postprocess(self, x, size):
        # x : NM C T V
        N, _, _, V, M = size
        _, c_new, T, _ = x.size()
        x = x.view(N, M, c_new, T, V)   # n,m,c,t,v
        x = x.permute(0, 3, 1, 4, 2).contiguous()   # n,t,m,v,c
        x = x.view(N, T, M*c_new*V)   # n,t,mvc
        x, a = self.transformer1(x)  # n,t,c
        x = x.mean(1)  # n,c
        return x, a


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model(torch.ones((1, 3, 300, 25, 2)))
