import math
import numpy as np
from typing import Optional

import torch.nn as nn

from model.aagcn import batch_norm_1d
from model.aagcn import bn_init
from model.aagcn import import_class
from model.aagcn import GCNUnit
from model.aagcn import TCNUnit


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
        y = self.tcn1(y)
        y = self.tse1(y)
        y = self.relu(y + self.residual(x))
        return y


# ------------------------------------------------------------------------------
# Network
# - change TCN to GCN style attention.
# - sct attention is removed.
# ------------------------------------------------------------------------------
class Model(nn.Module):
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
        super().__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.num_class = num_class

        self.data_bn = batch_norm_1d(num_person*in_channels*num_point,
                                     gbn_split)

        def _TCNGCNUnit(_in, _out, stride=1, residual=True):
            return TCNGCNUnit(_in,
                              _out,
                              A,
                              num_subset=num_subset,
                              stride=stride,
                              residual=residual,
                              adaptive=adaptive,
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

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()  # n,m,v,c,t
        x = x.view(N, -1, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()  # n,m,c,t,v  # noqa
        x = x.view(-1, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)
