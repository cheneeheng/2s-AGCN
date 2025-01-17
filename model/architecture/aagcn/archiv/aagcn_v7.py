import numpy as np
from typing import Optional

import torch
import torch.nn as nn

from model.architecture.aagcn.aagcn import import_class
from model.architecture.aagcn.aagcn import GCNUnit
from model.architecture.aagcn.aagcn import TCNUnit
from model.architecture.aagcn.aagcn import BaseModel


class TemporalSE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 coff_embedding: int = 4,
                 kernel_size: int = 9):
        super().__init__()
        pad = (kernel_size - 1) // 2
        if in_channels < coff_embedding * 2:
            inter_channels = in_channels
        else:
            inter_channels = in_channels//coff_embedding
        self.conv1 = nn.Conv1d(in_channels, inter_channels,
                               kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(inter_channels, 1,
                               kernel_size, padding=pad)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        se = x1.mean(-1)  # N C T
        se = self.conv1(se)
        se = self.relu(se)
        se = self.conv2(se)
        se = self.sigmoid(se)
        x = x2 * se.unsqueeze(-1) + x2
        return x


class AdaptiveGCN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: np.ndarray,
                 conv_d: nn.Conv2d,
                 num_subset: int = 3):
        super().__init__()
        self.num_subset = num_subset
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))  # Bk
        self.alpha = nn.Parameter(torch.zeros(1))  # G
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.tse1 = nn.ModuleList()
        for _ in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, out_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, out_channels, 1))
            self.tse1.append(TemporalSE(in_channels))
        self.soft = nn.Softmax(-2)
        self.conv_d = conv_d

    def forward(self, x):
        y = None
        N, C, T, V = x.size()
        A = self.PA  # Bk
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous()
            A1 = A1.view(N, V, -1)  # N V CT (theta)
            A2 = self.conv_b[i](x).view(N, -1, V)  # N CT V (phi)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A[i] + A1 * self.alpha
            A3 = x.view(N, -1, V)
            S1 = torch.matmul(A3, A1).view(N, C, -1, V)
            T1 = self.tse1[i](x, S1)
            z = self.conv_d[i](T1)
            y = z + y if y is not None else z
        return y


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
                 gbn_split: Optional[int] = None):
        super().__init__()
        self.gcn1 = GCNUnit(in_channels,
                            out_channels,
                            A,
                            num_subset=num_subset,
                            adaptive=adaptive,
                            attention=attention,
                            gbn_split=gbn_split)
        self.relu = nn.ReLU(inplace=True)

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

    def forward(self, x):
        y = self.gcn1(x)
        y = self.pool(y)
        y = y + self.residual(x)
        y = self.relu(y)
        return y


# ------------------------------------------------------------------------------
# Network
# - removed TCN, added TSE in the AdaptiveGCN.
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

        if adaptive:
            self.adaptive_fn = AdaptiveGCN

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

        self.init_fc(256, num_class)
