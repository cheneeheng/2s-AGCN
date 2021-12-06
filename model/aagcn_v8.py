import torch
from torch import nn

import numpy as np
from typing import Optional

from model.aagcn import import_class
from model.aagcn import TCNGCNUnit
from model.aagcn import BaseModel


class AdaptiveGCN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: np.ndarray,
                 conv_d: nn.Conv2d,
                 num_subset: int = 3,
                 num_splits: int = 5):
        super().__init__()
        self.num_subset = num_subset
        self.num_splits = num_splits
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))  # Bk
        self.alpha = nn.Parameter(torch.zeros(num_splits))  # G
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        for _ in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, out_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, out_channels, 1))
        self.soft = nn.Softmax(-2)
        self.conv_d = conv_d

    def forward(self, x):
        N, C, T, V = x.size()

        split_size = T // self.num_splits
        assert T % self.num_splits == 0
        x_tuple = torch.split(x, split_size, dim=2)

        A = self.PA  # Bk
        y = None

        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2)  # N V C T
            A2 = self.conv_b[i](x)  # N C T V
            A1 = torch.split(A1, split_size, dim=3)
            A2 = torch.split(A2, split_size, dim=2)
            A3 = []
            for j in range(self.num_splits):
                A1j = A1[j].contiguous().view(N, V, -1)  # N V Ct (theta)
                A2j = A2[j].contiguous().view(N, -1, V)  # N Ct V (phi)
                A12 = self.soft(torch.matmul(A1j, A2j) / A1j.size(-1))  # N V V
                A12 = A[i] + A12 * self.alpha[j]  # N V V
                A3j = x_tuple[j].contiguous().view(N, -1, V)  # N Ct V
                A3j = torch.matmul(A3j, A12).view(N, C, -1, V)  # N C t V
                A3.append(A3j)
            z = self.conv_d[i](torch.cat(A3, dim=2))
            assert z.shape[2] == T, z.shape[2]
            y = z + y if y is not None else z
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
