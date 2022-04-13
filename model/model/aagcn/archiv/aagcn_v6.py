import numpy as np
from typing import Optional

import torch.nn as nn

from model.model.aagcn.aagcn import import_class
from model.model.aagcn.aagcn import bn_init
from model.model.aagcn.aagcn import conv_init
from model.model.aagcn.aagcn import conv_branch_init
from model.model.aagcn.aagcn import batch_norm_2d
from model.model.aagcn.aagcn import AdaptiveGCN
from model.model.aagcn.aagcn import ChannelAttention
from model.model.aagcn.aagcn import SpatialAttention
from model.model.aagcn.aagcn import TemporalAttention
from model.model.aagcn.aagcn import TCNUnit
from model.model.aagcn.aagcn import BaseModel


class GCNUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: np.ndarray,
                 coff_embedding: int = 4,
                 num_subset: int = 3,
                 kernel_size_t: int = 9,
                 stride: int = 1,
                 adaptive: nn.Module = AdaptiveGCN,
                 attention: bool = True,
                 gbn_split: Optional[int] = None):
        super().__init__()

        def out_proj():
            pad = (kernel_size_t - 1) // 2
            return nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=(kernel_size_t, 1),
                             padding=(pad, 0),
                             stride=(stride, 1))

        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        num_jpts = A.shape[-1]

        self.conv_d = nn.ModuleList()
        for _ in range(self.num_subset):
            self.conv_d.append(out_proj())

        self.agcn = adaptive(in_channels,
                             inter_channels,
                             A,
                             self.conv_d,
                             num_subset=num_subset)

        if attention:
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            self.attn_s = SpatialAttention(out_channels, kernel_size=ker_jpt)
            self.attn_t = TemporalAttention(out_channels)
            self.attn_c = ChannelAttention(out_channels)
        else:
            self.attn_s, self.attn_t, self.attn_c = None, None, None

        # if the residual does not have the same channel dimensions.
        if in_channels != out_channels or stride > 1:
            self.down = nn.Sequential(
                out_proj(),
                batch_norm_2d(out_channels, gbn_split)
            )
        else:
            self.down = lambda x: x

        self.bn = batch_norm_2d(out_channels, gbn_split)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        y = self.agcn(x)
        y = self.bn(y) + self.down(x)
        y = self.relu(y)
        y = y if self.attn_s is None else self.attn_s(y)
        y = y if self.attn_t is None else self.attn_t(y)
        y = y if self.attn_c is None else self.attn_c(y)
        return y


class TGCNUnit(nn.Module):
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
                            stride=stride,
                            num_subset=num_subset,
                            adaptive=adaptive,
                            attention=attention,
                            gbn_split=gbn_split)

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
        y = self.gcn1(x) + self.residual(x)
        y = self.relu(y)
        return y


# ------------------------------------------------------------------------------
# Network
# - removed TCN. Added tcn conv in AdaptiveGCN (conv_d)
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
            return TGCNUnit(_in,
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
