import torch
from torch import nn
from torchinfo import summary

import numpy as np
from typing import Optional

from model.architecture.aagcn.aagcn import import_class
from model.architecture.aagcn.aagcn import TCNGCNUnit
from model.architecture.aagcn.aagcn import BaseModel


# ------------------------------------------------------------------------------
# Blocks
# Neural Machine Translation by Jointly Learning to Align and Translate
# relu / tanh
# ------------------------------------------------------------------------------
class FFNUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 inter_channels: int):
        super().__init__()
        self.l1 = nn.Linear(in_channels, inter_channels)
        self.l2 = nn.Linear(inter_channels, 1)
        # self.re = nn.ReLU()
        self.th = nn.Tanh()
        self.so = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        # x : N, L, C
        x = self.l2(self.th(self.l1(x)))
        # x = self.l2(self.re(self.l1(x)))
        x = self.so(x.squeeze(2))
        return x


# ------------------------------------------------------------------------------
# Network
# - uses gated mechanism (scalar attention)
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
        self.attn = FFNUnit(in_channels=256*num_point,
                            inter_channels=256)
        self.init_fc(256, num_class)

    def forward_postprocess(self, x, size):
        # x : NM C T V
        N, _, _, V, M = size
        _, c_new, t_new, _ = x.size()
        x = x.view(N*M, c_new, t_new, V)   # n,c,t,v
        x = x.permute(0, 2, 3, 1).contiguous()   # n,t,v,c
        x = x.view(N*M, t_new, V*c_new)   # n,t,vc
        a = self.attn(x)  # n,t
        x = torch.bmm(a.unsqueeze(1), x)  # n,1,vc
        x = x.view(N, M, V, c_new)   # n,m,v,c
        x = x.mean(2).mean(1)  # n,c
        return x, a


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model(torch.ones((1, 3, 300, 25, 2)))
