import torch
from torch import nn
from torchinfo import summary

import copy
import numpy as np
import math
from typing import Optional

from model.architecture.aagcn.aagcn import import_class
from model.architecture.aagcn.aagcn import conv_init
from model.architecture.aagcn.aagcn import bn_init
from model.architecture.aagcn.aagcn import batch_norm_2d
from model.architecture.aagcn.aagcn import GCNUnit
from model.architecture.aagcn.aagcn import AdaptiveGCN
from model.architecture.aagcn.aagcn import BaseModel


# ------------------------------------------------------------------------------
# AAGCN Modules
# ------------------------------------------------------------------------------
class TCNUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 9,
                 stride: int = 1,
                 pad: bool = True,
                 gbn_split: Optional[int] = None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if pad else 0
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(padding, 0),
                              stride=(stride, 1))
        self.bn = batch_norm_2d(out_channels, gbn_split)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        # relu is done after residual.
        x = self.conv(x)
        x = self.bn(x)
        return x


class TCNGCNUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: np.ndarray,
                 num_subset: int = 3,
                 kernel_size: int = 9,
                 stride: int = 1,
                 pad: bool = True,
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
                            kernel_size=kernel_size,
                            stride=stride,
                            pad=pad,
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
# Transformer Modules
# - pre-LN : On Layer Normalization in the Transformer Architecture
# - https: // pytorch.org/tutorials/beginner/transformer_tutorial.html
# ------------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.0,
                 max_len: int = 601):
        super().__init__()
        _pe = torch.empty(1, max_len, d_model)
        nn.init.normal_(_pe, std=0.02)  # bert
        self.pe = nn.Parameter(_pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : N, L, C
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x


class CosSinPositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.0,
                 max_len: int = 601):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : N, L, C
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x


class TransformerEncoderLayerExt(nn.TransformerEncoderLayer):
    """Option for pre of postnorm"""

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 pre_norm: bool = False,
                 A: np.ndarray = None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first)
        self.pre_norm = pre_norm
        if A is None:
            self.PA = None
        else:
            self.PA = nn.Parameter(
                torch.from_numpy(A.astype(np.float32)))  # Bk

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        if self.pre_norm:
            src = self.norm1(src)
            src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src1 = self.dropout(self.activation(self.linear1(src)))
            src2 = self.dropout2(self.linear2(src1))
            src = src + src2
            return src, attn
        else:
            src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src1 = self.dropout(self.activation(self.linear1(src)))
            src2 = self.dropout2(self.linear2(src1))
            src = src + src2
            src = self.norm2(src)
            return src, attn


class TransformerEncoderExt(nn.TransformerEncoder):

    def __init__(self, encoder_layer, num_layers, norm=None,
                 need_attn: bool = False):
        super().__init__(encoder_layer, num_layers, norm)
        self.need_attn = need_attn

    def forward(self,
                src: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        output = src
        attn_list = []
        for mod in self.layers:
            output, attn = mod(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask)
            if self.need_attn:
                attn_list.append(attn)
        if self.norm is not None:
            output = self.norm(output)
        return output, attn_list


# ------------------------------------------------------------------------------
# Network
# - uses original transformer
# - from v17
# n,mt,vc => temp trans (v17)
# nmt,v,c => spatial trans (individual)
# nt,mv,c => spatial trans (joint+individual)
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

                 kernel_size: int = 9,
                 pad: bool = True,

                 need_attn: bool = False,

                 trans_num_heads: int = 2,
                 trans_model_dim: int = 16,
                 trans_ffn_dim: int = 64,
                 trans_dropout: float = 0.2,
                 trans_activation: str = "gelu",
                 trans_prenorm: bool = False,
                 trans_num_layers: int = 1,

                 s_trans_num_heads: int = 2,
                 s_trans_model_dim: int = 16,
                 s_trans_ffn_dim: int = 64,
                 s_trans_dropout: float = 0.2,
                 s_trans_activation: str = "gelu",
                 s_trans_prenorm: bool = False,
                 s_trans_num_layers: int = 1,

                 add_A: bool = False,

                 pos_enc: str = 'True',
                 classifier_type: str = 'CLS',
                 model_layers: int = 10):
        super().__init__(num_class, num_point, num_person,
                         in_channels, drop_out, adaptive, gbn_split)

        self.need_attn = need_attn

        # 1. joint graph
        self.init_graph(graph, graph_args)

        # 2. aagcn layer
        def _TCNGCNUnit(_in, _out, stride=1, residual=True):
            return TCNGCNUnit(_in,
                              _out,
                              self.graph.A,
                              num_subset=num_subset,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              pad=pad,
                              residual=residual,
                              adaptive=self.adaptive_fn,
                              attention=attention,
                              gbn_split=gbn_split)

        self.init_model_backbone(model_layers=model_layers,
                                 tcngcn_unit=_TCNGCNUnit,
                                 output_channel=trans_model_dim)

        # 3. transformer (temporal)
        t_trans_dim = trans_model_dim*num_point
        t_trans_enc_layer = TransformerEncoderLayerExt(
            d_model=t_trans_dim,
            nhead=trans_num_heads,
            dim_feedforward=trans_ffn_dim*num_point,
            dropout=trans_dropout,
            activation=trans_activation,
            layer_norm_eps=1e-5,
            batch_first=True,
            pre_norm=trans_prenorm
        )
        self.t_trans_enc_layers = torch.nn.ModuleList(
            [copy.deepcopy(t_trans_enc_layer)
             for _ in range(trans_num_layers)])

        pos_enc = str(pos_enc)
        if pos_enc == 'True' or pos_enc == 'original':
            self.t_pos_encoder = PositionalEncoding(t_trans_dim)
        elif pos_enc == 'cossin':
            self.t_pos_encoder = CosSinPositionalEncoding(t_trans_dim)
        else:
            self.t_pos_encoder = lambda x: x

        self.classifier_type = classifier_type
        if classifier_type == 'CLS':
            self.cls_token = nn.Parameter(torch.randn(1, 1, t_trans_dim))
        else:
            self.cls_token = None

        # 4. transformer (spatial)
        s_trans_dim = s_trans_model_dim
        s_trans_enc_layer = TransformerEncoderLayerExt(
            d_model=s_trans_dim,
            nhead=s_trans_num_heads,
            dim_feedforward=s_trans_ffn_dim,
            dropout=s_trans_dropout,
            activation=s_trans_activation,
            layer_norm_eps=1e-5,
            batch_first=True,
            pre_norm=s_trans_prenorm,
            A=self.graph.A if add_A else None
        )
        self.s_trans_enc_layers = torch.nn.ModuleList(
            [copy.deepcopy(s_trans_enc_layer)
             for _ in range(s_trans_num_layers)])

        pos_enc = str(pos_enc)
        if pos_enc == 'True' or pos_enc == 'original':
            self.s_pos_encoder = PositionalEncoding(
                s_trans_dim, max_len=num_point)
        elif pos_enc == 'cossin':
            self.s_pos_encoder = CosSinPositionalEncoding(
                s_trans_dim, max_len=num_point)
        else:
            self.s_pos_encoder = lambda x: x

        # 5. classifier
        self.init_fc(t_trans_dim, num_class)

    def forward_postprocess(self, x: torch.Tensor, size: torch.Size):
        N, _, _, V, M = size
        _, C, T, _ = x.size()
        x = x.view(N, M, C, T, V).permute(0, 1, 3, 4, 2).contiguous()  # n,m,t,v,c  # noqa

        x = x.reshape(N*M*T, V, C)
        x = self.s_pos_encoder(x)

        x = x.reshape(N, M*T, V*C)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.t_pos_encoder(x)

        attn = [[], []]
        for s_layer, t_layer in zip(self.s_trans_enc_layers,
                                    self.t_trans_enc_layers):

            # x0 = x[:, 0:1, :]  # n,1,vc
            # x = x[:, 1:, :]  # n,mt,vc
            # x = x.reshape(N*M*T, V, C)
            # x, a = s_layer(x)
            # attn[0].append(a)

            # x = x.reshape(N, M*T, V*C)
            # x = torch.cat((x0, x), dim=1)
            # x, a = t_layer(x)
            # attn[1].append(a)

            x = x.reshape(-1, V, C)  # nmt,v,c
            A = s_layer.PA  # 3,v,v
            mask = None if A is None else A.repeat(N*(M*T+1), 1, 1)
            x, a = s_layer(x, mask)
            if self.need_attn:
                attn[0].append(a)

            x = x.reshape(N, -1, V*C)  # n,mt,vc
            x, a = t_layer(x)
            if self.need_attn:
                attn[1].append(a)

        if self.classifier_type == 'CLS':
            x = x[:, 0, :]  # n,vc
        elif self.classifier_type == 'GAP':
            x = x.mean(1)  # n,vc
        else:
            raise ValueError("Unknown classifier_type")

        return x, attn


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph, model_layers=101,
                  trans_num_layers=3,
                  trans_model_dim=24,
                  trans_ffn_dim=96,
                  trans_num_heads=3,
                  s_trans_num_layers=3,
                  s_trans_model_dim=24,
                  s_trans_ffn_dim=96,
                  s_trans_num_heads=3,
                  add_A=True,
                  kernel_size=3,
                  pad=False,
                  pos_enc='cossin'
                  )
    # print(model)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print([i[0] for i in model.named_parameters() if 'PA' in i[0]])
    model(torch.ones((3, 3, 300, 25, 2)))
