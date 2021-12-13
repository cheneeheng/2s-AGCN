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
                 lstm_in_channels: int,
                 proj_in_channels: int = 1,
                 proj_factor: int = 4,
                 num_layers: int = 1,
                 bidirectional: bool = False):
        super().__init__()

        if proj_factor > 1:
            self.proj = nn.Linear(
                proj_in_channels, proj_in_channels // proj_factor)
        else:
            self.proj = lambda x: x

        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=lstm_in_channels,
            hidden_size=lstm_in_channels // (bidirectional + 1),
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional,
            proj_size=0  # will be used for dim of h if > 0
        )

        self.norm = nn.LayerNorm(
            normalized_shape=lstm_in_channels,
            eps=1e-05,
            elementwise_affine=True
        )

    def forward(self, x: torch.Tensor, original_shape: bool = True):
        # x : N C T V
        # relu is done after residual.
        N, _, T, V = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()  # N T V C
        x = self.proj(x)
        x = x.view(N, T, -1)  # N T VC
        self.lstm.flatten_parameters()
        x, (hn, cn) = self.lstm(x)
        x = self.norm(x)  # N T VC
        if original_shape:
            x = x.view(N, T, V, -1)  # N T V C
            x = x.permute(0, 3, 1, 2).contiguous()  # N C T V
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
                 proj_factor: int = 1,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 postprocess_type: Optional[str] = 'GAP-TV',
                 model_layers: int = 10):
        super().__init__(num_class, num_point, num_person,
                         in_channels, drop_out, adaptive, gbn_split)

        assert proj_factor > 0

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

        self.init_model_backbone(model_layers=model_layers,
                                 tcngcn_unit=_TCNGCNUnit)

        self.rnn = LSTMUnit(
            lstm_in_channels=256*num_point // proj_factor,
            proj_in_channels=256,
            proj_factor=proj_factor,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )

        if self.postprocess_type == 'GAP-T':
            self.init_fc(256*num_point//proj_factor, num_class)
        elif self.postprocess_type == 'GAP-TV':
            self.init_fc(256//proj_factor, num_class)
        elif self.postprocess_type == 'LAST-T':
            self.init_fc(256*num_point//proj_factor, num_class)
        elif self.postprocess_type == 'LAST-TV':
            self.init_fc(256//proj_factor, num_class)

    def forward_postprocess(self, x: torch.Tensor, size: torch.Size):
        # x : n,c,t,v ; n=nm
        N, C, T, V, M = size
        t = x.size(2)
        if self.postprocess_type == 'GAP-T':
            x, hn, cn = self.rnn(x, False)  # n,t,vc
            x = x.view(N, M, t, V, -1)  # n,m,t,v,c
            x = x.mean(2).mean(1)  # n,c,v
            x = x.view(N, -1)  # n,cv
        elif self.postprocess_type == 'GAP-TV':
            x, hn, cn = self.rnn(x, True)  # n,c,t,v
            x = x.view(N, M, -1, t*V)  # n,m,c,tv
            x = x.mean(3).mean(1)  # n,c
        elif self.postprocess_type == 'LAST-T':
            x, hn, cn = self.rnn(x, False)  # n,t,vc
            x = x[:, -1, :]  # n,vc
            x = x.view(N, M, -1)  # n,m,vc
            x = x.mean(1)  # n,vc
        elif self.postprocess_type == 'LAST-TV':
            x, hn, cn = self.rnn(x, False)  # n,t,vc
            x = x[:, -1, :]  # n,vc
            x = x.view(N, M, V, -1)  # n,m,v,c
            x = x.mean(2).mean(1)  # n,c
        return x


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph, proj_factor=1, postprocess_type='GAP-TV',
                  bidirectional=False, num_layers=1)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model(torch.ones((1, 3, 300, 25, 2)))
