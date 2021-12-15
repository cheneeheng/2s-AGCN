import math
import numpy as np
from typing import Optional, Union

import torch
import torch.nn as nn
from torchinfo import summary

from model.ghostbatchnorm import GhostBatchNorm1d, GhostBatchNorm2d


def import_class(name: str):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------
def conv_branch_init(conv: Union[nn.Conv1d, nn.Conv2d], branches: int):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv: Union[nn.Conv1d, nn.Conv2d]):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn: Union[nn.BatchNorm1d, nn.BatchNorm2d], scale: int):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


# ------------------------------------------------------------------------------
# Blocks
# ------------------------------------------------------------------------------
def batch_norm_1d(num_channels: int, gbn_split: Optional[int] = None):
    if gbn_split is None or gbn_split < 2:
        return nn.BatchNorm1d(num_channels)
    else:
        return GhostBatchNorm1d(num_channels, gbn_split)


def batch_norm_2d(num_channels: int, gbn_split: Optional[int] = None):
    if gbn_split is None or gbn_split < 2:
        return nn.BatchNorm2d(num_channels)
    else:
        return GhostBatchNorm2d(num_channels, gbn_split)


class SpatialAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = 1,
                 kernel_size: int = 9):
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
    def __init__(self,
                 in_channels: int,
                 out_channels: int = 1,
                 kernel_size: int = 9):
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
    def __init__(self, in_channels: int, rr: int = 2):
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
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: np.ndarray,
                 conv_d: nn.Conv2d,
                 num_subset: int = 3):
        super().__init__()
        self.num_subset = num_subset
        self.A = torch.tensor(torch.from_numpy(A.astype(np.float32)),
                              requires_grad=False)
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
            A1 = A1.view(N, V, -1)  # N V CT (theta)
            A2 = self.conv_b[i](x).view(N, -1, V)  # N CT V (phi)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A[i] + A1 * self.alpha
            A3 = x.view(N, -1, V)
            z = self.conv_d[i](torch.matmul(A3, A1).view(N, C, -1, V))
            y = z + y if y is not None else z
        return y


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
class TCNUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 9,
                 stride: int = 1,
                 gbn_split: Optional[int] = None):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = batch_norm_2d(out_channels, gbn_split)

        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        # relu is done after residual.
        x = self.conv(x)
        x = self.bn(x)
        return x


class GCNUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: np.ndarray,
                 coff_embedding: int = 4,
                 num_subset: int = 3,
                 adaptive: nn.Module = AdaptiveGCN,
                 attention: bool = True,
                 gbn_split: Optional[int] = None):
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

        self.agcn = adaptive(in_channels, inter_channels,
                             A, self.conv_d, num_subset)

        if attention:
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            self.attn_s = SpatialAttention(out_channels, kernel_size=ker_jpt)
            self.attn_t = TemporalAttention(out_channels)
            self.attn_c = ChannelAttention(out_channels)
        else:
            self.attn_s, self.attn_t, self.attn_c = None, None, None

        # if the residual does not have the same channel dimensions.
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
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
        self.tcn1 = TCNUnit(out_channels,
                            out_channels,
                            stride=stride,
                            gbn_split=gbn_split)

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
        y = self.tcn1(y)
        y = y + self.residual(x)
        y = self.relu(y)
        return y


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------
class BaseModel(nn.Module):
    def __init__(self,
                 num_class: int = 60,
                 num_point: int = 25,
                 num_person: int = 2,
                 in_channels: int = 3,
                 drop_out: int = 0,
                 adaptive: bool = True,
                 gbn_split: Optional[int] = None,
                 fc_cv: bool = False):
        super().__init__()

        self.num_class = num_class

        self.graph = None

        self.adaptive_fn = AdaptiveGCN if adaptive else NonAdaptiveGCN

        self.data_bn = batch_norm_1d(num_person*in_channels*num_point,
                                     gbn_split)
        bn_init(self.data_bn, 1)

        self.l1 = None
        self.l2 = None
        self.l3 = None
        self.l4 = None
        self.l5 = None
        self.l6 = None
        self.l7 = None
        self.l8 = None
        self.l9 = None
        self.l10 = None

        self.fc = None
        self.fc_cv = fc_cv

        self.drop_out = nn.Dropout(drop_out) if drop_out else lambda x: x

    def init_model_backbone(self, model_layers, tcngcn_unit):
        if model_layers == 3:
            self.l1 = tcngcn_unit(3, 64, residual=False)
            self.l2 = lambda x: x
            self.l3 = lambda x: x
            self.l4 = lambda x: x
            self.l5 = tcngcn_unit(64, 128, stride=2)
            self.l6 = lambda x: x
            self.l7 = lambda x: x
            self.l8 = tcngcn_unit(128, 256, stride=2)
            self.l9 = lambda x: x
            self.l10 = lambda x: x
        elif model_layers == 6:
            self.l1 = tcngcn_unit(3, 64, residual=False)
            self.l2 = lambda x: x
            self.l3 = lambda x: x
            self.l4 = tcngcn_unit(64, 64)
            self.l5 = tcngcn_unit(64, 128, stride=2)
            self.l6 = lambda x: x
            self.l7 = tcngcn_unit(128, 128)
            self.l8 = tcngcn_unit(128, 256, stride=2)
            self.l9 = lambda x: x
            self.l10 = tcngcn_unit(256, 256)
        elif model_layers == 7:
            self.l1 = tcngcn_unit(3, 64, residual=False)
            self.l2 = lambda x: x
            self.l3 = tcngcn_unit(64, 64)
            self.l4 = tcngcn_unit(64, 64)
            self.l5 = tcngcn_unit(64, 128, stride=2)
            self.l6 = lambda x: x
            self.l7 = tcngcn_unit(128, 128)
            self.l8 = tcngcn_unit(128, 256, stride=2)
            self.l9 = lambda x: x
            self.l10 = tcngcn_unit(256, 256)
        elif model_layers == 10:
            self.l1 = tcngcn_unit(3, 64, residual=False)
            self.l2 = tcngcn_unit(64, 64)
            self.l3 = tcngcn_unit(64, 64)
            self.l4 = tcngcn_unit(64, 64)
            self.l5 = tcngcn_unit(64, 128, stride=2)
            self.l6 = tcngcn_unit(128, 128)
            self.l7 = tcngcn_unit(128, 128)
            self.l8 = tcngcn_unit(128, 256, stride=2)
            self.l9 = tcngcn_unit(256, 256)
            self.l10 = tcngcn_unit(256, 256)
        else:
            raise ValueError(
                f"Model with {model_layers} layers is not supported.")

    def init_fc(self, in_channels: int, out_channels: int):
        self.fc = nn.Linear(in_channels, out_channels)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / out_channels))

    def forward_preprocess(self, x, size):
        N, C, T, V, M = size
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # n,m,v,c,t
        x = x.view(N, -1, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()  # n,m,c,t,v  # noqa
        x = x.view(-1, C, T, V)  # nm,c,t,v
        return x

    def forward_model_backbone(self, x, size):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)  # nm,c,t,v
        return x

    def forward_postprocess(self, x, size):
        N, C, T, V, M = size
        c_new = x.size(1)
        if self.fc_cv:
            x = x.view(N, M, c_new, -1, V)  # n,m,c,t,v
            x = x.mean(3).mean(1)  # n,c,v
            x = x.view(N, -1)  # n,cv
        else:
            x = x.view(N, M, c_new, -1)  # n,m,c,tv
            x = x.mean(3).mean(1)  # n,c
        return x

    def forward_classifier(self, x, size):
        x = self.drop_out(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        size = x.size()
        x = self.forward_preprocess(x, size)
        x = self.forward_model_backbone(x, size)
        x = self.forward_postprocess(x, size)
        x = self.forward_classifier(x, size)
        return x


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
                 fc_cv: bool = False,
                 model_layers: int = 10):
        super().__init__(num_class, num_point, num_person,
                         in_channels, drop_out, adaptive, gbn_split, fc_cv)

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        def _TCNGCNUnit(in_channels, out_channels, stride=1, residual=True):
            return TCNGCNUnit(in_channels=in_channels,
                              out_channels=out_channels,
                              A=self.graph.A,
                              num_subset=num_subset,
                              stride=stride,
                              residual=residual,
                              adaptive=self.adaptive_fn,
                              attention=attention,
                              gbn_split=gbn_split)

        self.init_model_backbone(model_layers=model_layers,
                                 tcngcn_unit=_TCNGCNUnit)

        if fc_cv:
            self.init_fc(256*num_point, num_class)
        else:
            self.init_fc(256, num_class)


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph, fc_cv=False, model_layers=10, attention=True)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # model(torch.ones((1, 3, 300, 25, 2)))
    print(model)
