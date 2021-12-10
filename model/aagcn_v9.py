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
class LSTMUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_layers: int = 1,
                 bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=in_channels // (bidirectional + 1),
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional,
            proj_size=0  # will be used for dim of h if > 0
        )
        self.norm = nn.LayerNorm(
            normalized_shape=in_channels,
            eps=1e-05,
            elementwise_affine=True
        )

    def forward(self, x: torch.Tensor, original_shape: bool = True):
        # x : N C T V
        # relu is done after residual.
        N, _, T, V = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # N T C V
        x = x.view(N, T, -1)  # N T CV
        x, (hn, cn) = self.lstm(x)
        x = self.norm(x)  # N T CV
        if original_shape:
            x = x.view(N, T, -1, V)  # N T C V
            x = x.permute(0, 2, 1, 3).contiguous()  # N C T V
        return x, hn, cn


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
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 postprocess_type: Optional[str] = 'GAP-TV'):
        super().__init__(num_class, num_point, num_person,
                         in_channels, drop_out, adaptive, gbn_split)

        assert postprocess_type in ['GAP-T', 'GAP-TV', 'LAST-T', 'LAST-TV']
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
        elif self.postprocess_type == 'LAST-T':
            self.init_fc(256*num_point, num_class)
        elif self.postprocess_type == 'LAST-TV':
            self.init_fc(256, num_class)

        self.rnn = LSTMUnit(
            in_channels=256*num_point,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )

    def forward_postprocess(self, x: torch.Tensor, size: torch.Size):
        # x : NM C T V
        N, C, T, V, M = size
        c_new = x.size(1)
        if self.postprocess_type == 'GAP-T':
            x, hn, cn = self.rnn(x, False)  # N T CV
            x = x.view(N, M, -1, c_new, V)  # n,m,t,c,v
            x = x.mean(2).mean(1)  # n,c,v
            x = x.view(N, -1)  # n,cv
        elif self.postprocess_type == 'GAP-TV':
            x, hn, cn = self.rnn(x, True)  # N C T V
            x = x.view(N, M, c_new, -1)  # n,m,c,tv
            x = x.mean(3).mean(1)  # n,c
        elif self.postprocess_type == 'LAST-T':
            x, hn, cn = self.rnn(x, False)  # N T CV
            x = x[:, -1, :]  # N CV
            x = x.view(N, M, -1)  # n,m,cv
            x = x.mean(1)  # n,cv
        elif self.postprocess_type == 'LAST-TV':
            x, hn, cn = self.rnn(x, False)  # N T CV
            x = x[:, -1, :]  # N CV
            x = x.view(N, M, c_new, V)  # n,m,c,v
            x = x.mean(3).mean(1)  # n,c
        return x


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model(torch.ones((1, 3, 300, 25, 2)))
