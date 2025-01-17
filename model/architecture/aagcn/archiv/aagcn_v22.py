import torch
from torch import nn
from torchinfo import summary

import numpy as np
import math
from typing import Optional, List

from model.architecture.aagcn.aagcn import import_class
from model.architecture.aagcn.aagcn import conv_init
from model.architecture.aagcn.aagcn import bn_init
from model.architecture.aagcn.aagcn import batch_norm_2d
from model.architecture.aagcn.aagcn import GCNUnit
from model.architecture.aagcn.aagcn import AdaptiveGCN
from model.architecture.aagcn.aagcn import BaseModel


# ------------------------------------------------------------------------------
# Modules
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
# Blocks
# - pre-LN : On Layer Normalization in the Transformer Architecture
# - https: // pytorch.org/tutorials/beginner/transformer_tutorial.html
# ------------------------------------------------------------------------------
class PositionalEncodingBase(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.0,
                 max_len: int = 601):
        super().__init__()
        self.pe = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : N, L, C
        if self.pe is None:
            x = x + getattr(self, self.buffer_name)[:, :x.size(1), :]
        else:
            x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x


class PositionalEncoding(PositionalEncodingBase):
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.0,
                 max_len: int = 601):
        super().__init__(d_model, dropout, max_len)
        _pe = torch.empty(1, max_len, d_model)
        nn.init.normal_(_pe, std=0.02)  # bert
        self.pe = nn.Parameter(_pe)


class CosSinPositionalEncoding(PositionalEncodingBase):
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.0,
                 max_len: int = 601,
                 name: str = ''):
        super().__init__(d_model, dropout, max_len)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe' + name, pe)
        self.buffer_name = 'pe' + name


class PositionalEncoding2D(nn.Module):
    def __init__(self,
                 d_p: Optional[int] = None,
                 dropout: float = 0.0,
                 length: int = 101):
        super().__init__()
        if d_p is None or d_p == 0:
            _pe = torch.empty(length, length)
            nn.init.normal_(_pe, std=0.02)  # bert
            self.pe = nn.Parameter(_pe)
        else:
            _peq = torch.empty(length, d_p)
            _pek = torch.empty(length, d_p)
            nn.init.normal_(_peq, std=0.02)  # bert
            nn.init.normal_(_pek, std=0.02)  # bert
            self.peq = nn.Parameter(_peq)
            self.pek = nn.Parameter(_pek)
            self.pe = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        # # x : N, L, L
        # x = x + self.pe
        # x = self.dropout(x)
        # return x
        if self.pe is None:
            return torch.matmul(self.peq, self.pek.transpose(-2, -1))
        else:
            return self.pe


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
            src2 = self.linear2(self.dropout(
                self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src, attn


class TransformerEncoderExt(nn.TransformerEncoder):

    def __init__(self,
                 encoder_layer,
                 num_layers,
                 norm=None,
                 need_attn: bool = False):
        super().__init__(encoder_layer, num_layers, norm)
        self.need_attn = need_attn

    def forward(self,
                src: torch.Tensor,
                mask: Optional[List[torch.Tensor]] = None,
                pe_list: Optional[nn.ModuleList] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        output = src

        attn_list = []

        if mask is None:
            mask = [None for _ in range(self.num_layers)]

        for mod, mask_i, pos_encoder in zip(self.layers, mask, pe_list):
            output = pos_encoder(output)
            output, attn = mod(output, src_mask=mask_i,
                               src_key_padding_mask=src_key_padding_mask)
            if self.need_attn:
                attn_list.append(attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_list


def generate_square_subsequent_mask(sz: int, device: str) -> torch.Tensor:
    """From nn.Transformer"""
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# ------------------------------------------------------------------------------
# Network
# - Add PE to each trans layer.
# - from v17
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
                 attn_masking: Optional[dict] = None,

                 trans_num_heads: int = 2,
                 trans_model_dim: int = 16,
                 trans_ffn_dim: int = 64,
                 trans_dropout: float = 0.2,
                 trans_activation: str = "gelu",
                 trans_prenorm: bool = False,
                 trans_num_layers: int = 1,

                 pos_enc: str = 'True',
                 classifier_type: str = 'CLS',
                 model_layers: int = 10):
        super().__init__(num_class, num_point, num_person,
                         in_channels, drop_out, adaptive, gbn_split)

        self.attn_mask = None
        self.trans_num_heads = trans_num_heads
        self.kernel_size = kernel_size

        self.init_graph(graph, graph_args)

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

        trans_dim = trans_model_dim*num_point

        pos_enc = str(pos_enc)
        if pos_enc == 'True' or pos_enc == 'original':
            self.pos_encoder = nn.ModuleList([
                PositionalEncoding(trans_dim)
                for _ in range(trans_num_layers)
            ])
        elif pos_enc == 'cossin':
            self.pos_encoder = nn.ModuleList([
                CosSinPositionalEncoding(trans_dim, name=str(i))
                for i in range(trans_num_layers)
            ])
        else:
            self.pos_encoder = [lambda x: x for _ in range(trans_num_layers)]

        self.classifier_type = classifier_type
        if classifier_type == 'CLS':
            self.cls_token = nn.Parameter(torch.randn(1, 1, trans_dim))
        else:
            self.cls_token = None

        trans_enc_layer = TransformerEncoderLayerExt(
            d_model=trans_dim,
            nhead=trans_num_heads,
            dim_feedforward=trans_ffn_dim*num_point,
            dropout=trans_dropout,
            activation=trans_activation,
            layer_norm_eps=1e-5,
            batch_first=True,
            pre_norm=trans_prenorm
        )

        self.trans_enc = TransformerEncoderExt(
            trans_enc_layer,
            num_layers=trans_num_layers,
            norm=None,
            need_attn=need_attn
        )

        self.init_fc(trans_dim, num_class)

    def forward_postprocess(self, x: torch.Tensor, size: torch.Size):
        N, _, _, V, M = size
        _, C, T, _ = x.size()
        x = x.view(N, M, C, T, V).permute(0, 1, 3, 4, 2).contiguous()  # n,m,t,v,c  # noqa
        x = x.view(N, M*T, C*V)  # n,mt,vc

        if self.cls_token is not None:
            cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
            x = torch.cat((cls_tokens, x), dim=1)
        # x = self.pos_encoder(x)

        x, attn = self.trans_enc(x, self.attn_mask, self.pos_encoder)
        if self.classifier_type == 'CLS':
            x = x[:, 0, :]  # n,vc
        elif self.classifier_type == 'GAP':
            x = x.mean(1)  # n,vc
        else:
            raise ValueError("Unknown classifier_type")

        return x, attn


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph,
                  model_layers=101,
                  trans_num_layers=3,
                  kernel_size=3,
                  pad=False,
                  pos_enc='cossin',
                  attn_masking={'d_p': 8, 'dropout': 0, 'length': 101},
                  )
    # print(model)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    x = torch.ones((5, 3, 300, 25, 2))
    x[:, :, 200:, :, :] = 0.0
    model(x)
