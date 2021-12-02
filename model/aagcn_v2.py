import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from .ghostbatchnorm import GhostBatchNorm1d, GhostBatchNorm2d


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------
def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


# ------------------------------------------------------------------------------
# Blocks
# ------------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels=1, kernel_size=9):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv_sa = nn.Conv1d(in_channels, out_channels, kernel_size,
                                 padding=pad)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = x.mean(-2)  # N C V
        se = self.sigmoid(self.conv_sa(se))
        x = x * se.unsqueeze(-2) + x
        return x


class TemporalAttention(nn.Module):
    def __init__(self, in_channels, out_channels=1, kernel_size=9):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv_ta = nn.Conv1d(in_channels, out_channels, kernel_size,
                                 padding=pad)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = x.mean(-1)  # N C T
        se = self.sigmoid(self.conv_ta(se))
        x = x * se.unsqueeze(-1) + x
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, rr=2):
        super().__init__()
        self.fc1c = nn.Linear(in_channels, in_channels // rr)
        self.fc2c = nn.Linear(in_channels // rr, in_channels)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        se = x.mean(-1).mean(-1)  # N C
        se = self.relu(self.fc1c(se))
        se = self.sigmoid(self.fc2c(se))
        x = x * se.unsqueeze(-1).unsqueeze(-1) + x
        return x


class NonAdaptiveGCN(nn.Module):
    def __init__(self, A, conv_d, num_subset=3):
        super().__init__()
        self.num_subset = num_subset
        self.A = Variable(torch.from_numpy(
            A.astype(np.float32)), requires_grad=False)
        self.conv_d = conv_d

    def forward(self, x):
        N, C, T, V = x.size()
        y = None
        # A = self.A.cuda(x.get_device())
        A = self.A
        for i in range(self.num_subset):
            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        return y


class AdaptiveGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, conv_d, num_subset=3):
        super().__init__()
        self.num_subset = num_subset
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))  # Bk
        self.alpha = nn.Parameter(torch.zeros(1))  # G
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        for _ in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, out_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, out_channels, 1))
        self.soft = nn.Softmax(-2)
        self.conv_d = conv_d

    def forward(self, x):
        y = None
        N, C, T, V = x.size()
        A = self.PA  # Bk
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous()
            A1 = A1.view(N, V, -1)
            A2 = self.conv_b[i](x).view(N, -1, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A[i] + A1 * self.alpha
            A3 = x.view(N, -1, V)
            z = self.conv_d[i](torch.matmul(A3, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        return y


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
class TCNUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1,
                 gbn_split=None):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0),
                              stride=(stride, 1))
        if gbn_split is None:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = GhostBatchNorm2d(out_channels, gbn_split)

        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        # relu is done after residual.
        x = self.conv(x)
        x = self.bn(x)
        return x


class GCNUnit(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4,
                 num_subset=3, adaptive=True, attention=True,
                 gbn_split=None):
        super().__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        num_jpts = A.shape[-1]

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if adaptive:
            self.agcn = AdaptiveGCN(
                in_channels, inter_channels, A, self.conv_d, num_subset)
        else:
            self.agcn = NonAdaptiveGCN(A, self.conv_d, num_subset)

        self.attention = attention
        if attention:
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            self.attn_s = SpatialAttention(out_channels, kernel_size=ker_jpt)
            self.attn_t = TemporalAttention(out_channels)
            self.attn_c = ChannelAttention(out_channels)

        # if the residual does not have the same channel dimensions.
        if in_channels != out_channels:
            if gbn_split is None:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    GhostBatchNorm2d(out_channels)
                )
        else:
            self.down = lambda x: x

        if gbn_split is None:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = GhostBatchNorm2d(out_channels)

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

        if self.attention:
            y = self.attn_s(y)
            y = self.attn_t(y)
            y = self.attn_c(y)

        return y


class TCNGCNUnit(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_subset=3, stride=1,
                 residual=True, adaptive=True, attention=True,
                 gbn_split=None):
        super().__init__()
        self.gcn1 = GCNUnit(in_channels, out_channels, A,
                            num_subset=num_subset,
                            adaptive=adaptive, attention=attention,
                            gbn_split=gbn_split)
        self.tcn1 = TCNUnit(out_channels, out_channels, stride=stride,
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
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


# ------------------------------------------------------------------------------
# Network
# - A does not depend on predefined A matrix.
# ------------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2,
                 num_subset=3, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, attention=True,
                 gbn_split=None):
        super().__init__()

        # if graph is None:
        #     raise ValueError()
        # else:
        #     Graph = import_class(graph)
        #     self.graph = Graph(**graph_args)

        # A = self.graph.A

        A = np.ones((num_subset, num_point, num_point))
        self.num_class = num_class

        if gbn_split is None:
            self.data_bn = nn.BatchNorm1d(num_person*in_channels*num_point)
        else:
            self.data_bn = GhostBatchNorm1d(
                num_person*in_channels*num_point, gbn_split)

        def _TCNGCNUnit(_in, _out, stride=1, residual=True):
            return TCNGCNUnit(_in, _out, A, num_subset=num_subset,
                              stride=stride, residual=residual,
                              adaptive=adaptive, attention=attention,
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