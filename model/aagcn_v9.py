import torch
from torch import nn
from torchinfo import summary

import numpy as np
from typing import Optional
import graph

from model.aagcn import import_class
from model.aagcn import batch_norm_2d
from model.aagcn import AdaptiveGCN
from model.aagcn import GCNUnit
from model.aagcn import TCNUnit
from model.aagcn import TCNGCNUnit as TCNGCNUnitOri
from model.aagcn import BaseModel


# ------------------------------------------------------------------------------
# Blocks
# ------------------------------------------------------------------------------
class LSTMUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 gbn_split: Optional[int] = None):
        super().__init__()
        # __init__(self, input_size, hidden_size, num_layers, num_classes):
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=in_channels // (bidirectional + 1),
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional,
            proj_size=0)  # will be used for dim of h if > 0
        self.bn = batch_norm_2d(out_channels, gbn_split)

    def forward(self, x):
        # x : N C T V
        # relu is done after residual.
        N, _, T, V = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # N T C V
        x = x.view(N, T, -1)  # N T CV
        x, (hn, cn) = self.lstm(x)
        x = x.view(N, T, -1, V)  # N T C V
        x = x.permute(0, 2, 1, 3).contiguous()  # N C T V
        x = self.bn(x)
        return x


class TCNGCNUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: np.ndarray,
                 num_subset: int = 3,
                 stride: int = 1,
                 residual: bool = True,
                 adaptive: nn.Module = AdaptiveGCN,
                 attention: bool = True,
                 gbn_split: Optional[int] = None,
                 num_point: int = 25,
                 num_layers: int = 1,
                 bidirectional: bool = False):
        super().__init__()
        self.gcn1 = GCNUnit(in_channels,
                            out_channels,
                            A,
                            num_subset=num_subset,
                            adaptive=adaptive,
                            attention=attention,
                            gbn_split=gbn_split)
        self.rnn1 = LSTMUnit(out_channels * num_point,
                             out_channels,
                             num_layers=num_layers,
                             bidirectional=bidirectional,
                             gbn_split=gbn_split)

        if stride > 1:
            self.pool = nn.AvgPool2d((stride, 1))
        else:
            self.pool = lambda x: x

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            # if the residual does not have the same channel dimensions.
            # if stride > 1
            self.residual = TCNUnit(in_channels,
                                    out_channels,
                                    kernel_size=1,
                                    stride=stride,
                                    gbn_split=gbn_split)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.gcn1(x)
        y = self.rnn1(y)
        y = self.pool(y)
        y = y + self.residual(x)
        y = self.relu(y)
        return y


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
                 bidirectional: bool = False):
        super().__init__(num_class, num_point, num_person,
                         in_channels, drop_out, adaptive, gbn_split)

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        def _TCNGCNUnitOri(_in, _out, stride=1, residual=True):
            return TCNGCNUnitOri(_in,
                                 _out,
                                 self.graph.A,
                                 num_subset=num_subset,
                                 stride=stride,
                                 residual=residual,
                                 adaptive=self.adaptive_fn,
                                 attention=attention,
                                 gbn_split=gbn_split)

        def _TCNGCNUnit(_in, _out, stride=1, residual=True):
            return TCNGCNUnit(_in,
                              _out,
                              self.graph.A,
                              num_subset=num_subset,
                              stride=stride,
                              residual=residual,
                              adaptive=self.adaptive_fn,
                              attention=attention,
                              gbn_split=gbn_split,
                              num_point=num_point,
                              num_layers=num_layers,
                              bidirectional=bidirectional)

        # self.l1 = _TCNGCNUnit(3, 64, residual=False)
        # self.l2 = lambda x: x
        # self.l3 = lambda x: x
        # self.l4 = lambda x: x
        # self.l5 = lambda x: x
        # self.l6 = lambda x: x
        # self.l7 = lambda x: x
        # self.l8 = lambda x: x
        # self.l9 = lambda x: x
        # self.l10 = lambda x: x

        self.l1 = _TCNGCNUnitOri(3, 64, residual=False)
        self.l2 = _TCNGCNUnitOri(64, 64)
        self.l3 = _TCNGCNUnitOri(64, 64)
        self.l4 = _TCNGCNUnitOri(64, 64)
        self.l5 = _TCNGCNUnitOri(64, 128, stride=2)
        self.l6 = _TCNGCNUnitOri(128, 128)
        self.l7 = _TCNGCNUnitOri(128, 128)
        self.l8 = _TCNGCNUnitOri(128, 256, stride=2)
        self.l9 = _TCNGCNUnitOri(256, 256)
        self.l10 = _TCNGCNUnit(256, 256)


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # model(torch.ones((1, 3, 300, 25, 2)))
