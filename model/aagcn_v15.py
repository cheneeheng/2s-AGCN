import torch
from torch import nn
from torchinfo import summary

import numpy as np
import math
from typing import Optional

from model.aagcn import import_class
from model.aagcn import AdaptiveGCN
from model.aagcn import GCNUnit
from model.aagcn import TCNUnit
from model.aagcn import BaseModel


# ------------------------------------------------------------------------------
# Blocks
# - pre-LN : On Layer Normalization in the Transformer Architecture
# - https: // pytorch.org/tutorials/beginner/transformer_tutorial.html
# ------------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.0,
                 max_len: int = 301):
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
                 pre_norm: bool = False) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first)
        self.pre_norm = pre_norm

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        if self.pre_norm:
            src = self.norm1(src)
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src1 = self.dropout(self.activation(self.linear1(src)))
            src2 = self.dropout2(self.linear2(src1))
            src = src + src2
            return src
        else:
            return super().forward(src, src_mask, src_key_padding_mask)


class TransformerUnit(nn.Module):
    def __init__(self,
                 num_point: int = 25,
                 trans_num_heads: int = 2,
                 trans_model_dim: int = 16,
                 trans_ffn_dim: int = 64,
                 trans_dropout: float = 0.2,
                 trans_activation: str = "gelu",
                 trans_prenorm: bool = False,
                 trans_num_layers: int = 1,
                 trans_length: int = 9,
                 pos_enc: bool = True,
                 stride: int = 1,
                 #  classifier_type: str = 'CLS',
                 ):
        super().__init__()

        if pos_enc:
            self.pos_encoder = PositionalEncoding(trans_model_dim*num_point)
        else:
            self.pos_encoder = lambda x: x

        trans_enc_layer = TransformerEncoderLayerExt(
            d_model=trans_model_dim*num_point,
            nhead=trans_num_heads,
            dim_feedforward=trans_ffn_dim*num_point,
            dropout=trans_dropout,
            activation=trans_activation,
            layer_norm_eps=1e-5,
            batch_first=True,
            pre_norm=trans_prenorm
        )

        self.trans_enc = nn.TransformerEncoder(
            trans_enc_layer,
            num_layers=trans_num_layers,
            norm=None
        )

        self.stride = stride
        self.trans_length = trans_length

    def forward(self, x):
        # x : nm,c,t,v
        N, C, T, V = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()  # n,t,v,c  # noqa
        x = x.view(N, T, C*V)  # n,t,vc

        x = self.pos_encoder(x)  # n,t,vc

        x_t = torch.zeros((N, T//self.stride, C*V))
        for i in range(0,
                       x.size(1)-(self.trans_length*self.stride),
                       self.stride):
            x_i = torch.zeros((N, T//self.stride, C*V))
            x_i[:, i//2:i//2+self.trans_length, :] = \
                self.trans_enc(x[:, i:i+self.trans_length, :])
            x_t += x_i

        x = x_t.view(N, T//self.stride, V, C)  # n,t,v,c
        x = x.permute(0, 3, 1, 2).contiguous()  # n,t,v,c  # noqa
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
                 num_point: int = 25,
                 trans_num_heads: int = 2,
                 trans_dropout: float = 0.2,
                 trans_activation: str = "gelu",
                 trans_prenorm: bool = False,
                 trans_num_layers: int = 1,
                 trans_length: int = 9,
                 pos_enc: bool = True,
                 ):
        super().__init__()
        self.gcn1 = GCNUnit(in_channels,
                            out_channels,
                            A,
                            num_subset=num_subset,
                            adaptive=adaptive,
                            attention=attention,
                            gbn_split=gbn_split
                            )
        self.tcn1 = TransformerUnit(num_point=num_point,
                                    trans_num_heads=trans_num_heads,
                                    trans_model_dim=out_channels,
                                    trans_ffn_dim=out_channels*4,
                                    trans_dropout=trans_dropout,
                                    trans_activation=trans_activation,
                                    trans_prenorm=trans_prenorm,
                                    trans_num_layers=trans_num_layers,
                                    trans_length=trans_length,
                                    pos_enc=pos_enc,
                                    stride=stride
                                    )

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
# - uses original transformer to replace tcn
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
                 trans_num_heads: int = 2,
                 trans_model_dim: int = 16,
                 trans_ffn_dim: int = 64,
                 trans_dropout: float = 0.2,
                 trans_activation: str = "gelu",
                 trans_prenorm: bool = False,
                 trans_num_layers: int = 1,
                 trans_length: int = 9,
                 pos_enc: bool = True,
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
                              trans_num_heads=trans_num_heads,
                              trans_dropout=trans_dropout,
                              trans_activation=trans_activation,
                              trans_prenorm=trans_prenorm,
                              trans_num_layers=trans_num_layers,
                              trans_length=trans_length,
                              pos_enc=pos_enc,
                              )

        self.init_model_backbone(model_layers=model_layers,
                                 tcngcn_unit=_TCNGCNUnit,
                                 output_channel=trans_model_dim)

        self.init_fc(256, num_class)


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph,
                  model_layers=3,
                  trans_num_layers=1,
                  trans_model_dim=64,
                  trans_ffn_dim=256,
                  )
    # print(model)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model(torch.ones((1, 3, 300, 25, 2)))
