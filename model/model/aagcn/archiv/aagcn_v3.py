import numpy as np
from typing import Optional

import torch
import torch.nn as nn


from model.model.aagcn.aagcn import import_class
from model.model.aagcn.aagcn import TCNGCNUnit
from model.model.aagcn.aagcn import BaseModel


class AdaptiveGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, conv_d, num_subset=3):
        super().__init__()
        self.num_subset = num_subset
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))  # Bk
        self.alpha = nn.Parameter(torch.zeros(1))  # G
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_c = nn.ModuleList()
        for _ in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, out_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, out_channels, 1))
            self.conv_c.append(nn.Conv2d(in_channels, in_channels, 1))
        self.soft = nn.Softmax(-2)
        self.conv_d = conv_d

    def forward(self, x):
        y = None
        N, C, T, V = x.size()
        A = self.PA  # Bk
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous()
            A2 = self.conv_b[i](x)
            A3 = self.conv_c[i](x)
            A1 = A1.view(N, V, -1)  # N V C'T (theta)
            A2 = A2.view(N, -1, V)  # N C'T V (phi)
            A3 = A3.view(N, -1, V)  # N CT V
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A[i] + A1 * self.alpha
            z = self.conv_d[i](torch.matmul(A3, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        return y


# ------------------------------------------------------------------------------
# Network
# - Added additional projection in the GCN. > no noticable change.
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
