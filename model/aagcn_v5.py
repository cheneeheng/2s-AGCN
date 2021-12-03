import numpy as np
from typing import Optional

import torch.nn as nn

from model.aagcn import import_class
from model.aagcn import GCNUnit
from model.aagcn import TCNUnit
from model.aagcn import BaseModel


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, coff_embedding: int = 4):
        super().__init__()
        self.fc1c = nn.Linear(in_channels, in_channels // coff_embedding)
        self.fc2c = nn.Linear(in_channels // coff_embedding, in_channels)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.kaiming_normal_(self.fc2c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.bias, 0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x : N C T V
        se = x.mean(-1).mean(-1)  # N C
        se = self.fc1c(se)
        se = self.relu(se)
        se = self.fc2c(se)
        se = self.sigmoid(se)
        x = x * se.unsqueeze(-1).unsqueeze(-1) + x
        return x


class TCNGCNUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: np.ndarray,
                 num_subset: int = 3,
                 stride: int = 1,
                 residual: bool = True,
                 adaptive: bool = True,
                 attention: bool = True,
                 gbn_split: Optional[int] = None):
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
                            gbn_split=gbn_split)
        self.tse1 = SqueezeExcitation(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            # if the residual does not have the same channel dimensions.
            # if stride > 1
            self.residual = TCNUnit(in_channels, out_channels,
                                    kernel_size=1, stride=stride,
                                    gbn_split=gbn_split)

    def forward(self, x):
        y = self.gcn1(x)
        y = self.tcn1(y)  # N C T V
        y = y.permute(0, 2, 1, 3).contiguous()  # N T C V
        y = self.tse1(y)
        y = y.permute(0, 1, 2, 3).contiguous()  # N C T V
        y = self.relu(y + self.residual(x))
        return y


# ------------------------------------------------------------------------------
# Network
# - added se block after TCN, sct attention is removed.
# - without the tse block.
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
                 gbn_split: Optional[int] = None):
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
