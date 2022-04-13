import torch
from torch import nn
from torchinfo import summary

import numpy as np
import math
from typing import Optional

from model.model.aagcn.aagcn import import_class
from model.model.aagcn.aagcn import bn_init
from model.model.aagcn.aagcn import conv_init
from model.model.aagcn.aagcn import batch_norm_2d
from model.model.aagcn.aagcn import AdaptiveGCN
from model.model.aagcn.aagcn import GCNUnit
from model.model.aagcn.aagcn import BaseModel


# ------------------------------------------------------------------------------
# Blocks
# ------------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_point: int,
                 dropout: float = 0.0,
                 max_len: int = 300):
        super().__init__()
        _pe = torch.empty(1, d_model, max_len, num_point)
        nn.init.normal_(_pe, std=0.02)  # bert
        self.pe = nn.Parameter(_pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : n,c,t,v
        x = x + self.pe[:, :, :x.size(2), :]
        x = self.dropout(x)
        return x


class TCNUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 9,
                 stride: int = 1,
                 pos_enc: bool = False,
                 num_point: int = 25,
                 gbn_split: Optional[int] = None):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = batch_norm_2d(out_channels, gbn_split)

        if pos_enc:
            self.pos_encoder = PositionalEncoding(out_channels, num_point)
        else:
            self.pos_encoder = lambda x: x

        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        # relu is done after residual.
        x = self.pos_encoder(x)  # n,c,t,v
        x = self.conv(x)
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
                 pos_enc: bool = False,
                 num_point: int = 25,
                 ):
        super().__init__()
        self.gcn1 = GCNUnit(in_channels,
                            out_channels,
                            A,
                            num_subset=num_subset,
                            adaptive=adaptive,
                            attention=attention,
                            gbn_split=gbn_split)
        self.tcn1 = TCNUnit(out_channels,
                            out_channels,
                            stride=stride,
                            gbn_split=gbn_split,
                            num_point=num_point,
                            pos_enc=pos_enc)

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
                                    gbn_split=gbn_split,
                                    num_point=num_point,
                                    pos_enc=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.gcn1(x)
        y = self.tcn1(y)
        y = y + self.residual(x)
        y = self.relu(y)
        return y


# ------------------------------------------------------------------------------
# Network
# - puts positional encoder into tcn
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
                 pos_enc: bool = False,
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
                              gbn_split=gbn_split,
                              num_point=num_point,
                              pos_enc=pos_enc,
                              )

        self.init_model_backbone(model_layers=model_layers,
                                 tcngcn_unit=_TCNGCNUnit)

        self.init_fc(256, num_class)


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph,
                  model_layers=10,
                  pos_enc=True
                  )
    # print(model)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model(torch.ones((1, 3, 300, 25, 2)))
