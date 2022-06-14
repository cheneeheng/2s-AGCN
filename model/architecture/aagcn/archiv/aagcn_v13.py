import torch
from torch import nn
from torchinfo import summary

import numpy as np
import math
from typing import Optional

from model.architecture.aagcn.aagcn import import_class
from model.architecture.aagcn.aagcn import TCNGCNUnit
from model.architecture.aagcn.aagcn import BaseModel


# ------------------------------------------------------------------------------
# Blocks
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


# ------------------------------------------------------------------------------
# Network
# - uses original transformer
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
                 projection_layer: bool = True,
                 trans_num_heads: int = 2,
                 trans_model_dim: int = 16,
                 trans_ffn_dim: int = 64,
                 trans_dropout: float = 0.2,
                 trans_activation: str = "gelu",
                 trans_prenorm: bool = False,
                 trans_num_layers: int = 1,
                 pos_enc: bool = True,
                 classifier_type: str = 'CLS',
                 model_layers: int = 10):
        super().__init__(num_class, num_point, num_person,
                         in_channels, drop_out, adaptive, gbn_split)

        self.projection_layer = projection_layer
        if projection_layer:
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
                                  gbn_split=gbn_split)

            self.init_model_backbone(model_layers=model_layers,
                                     tcngcn_unit=_TCNGCNUnit,
                                     output_channel=trans_model_dim)

        if pos_enc:
            self.pos_encoder = PositionalEncoding(trans_model_dim*num_point)
        else:
            self.pos_encoder = lambda x: x

        self.classifier_type = classifier_type
        if classifier_type == 'CLS':
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, trans_model_dim*num_point))
        else:
            self.cls_token = None

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

        self.init_fc(trans_model_dim*num_point, num_class)

    def forward_postprocess(self, x: torch.Tensor, size: torch.Size):
        N, _, _, V, M = size
        _, C, T, _ = x.size()
        x = x.view(N, M, C, T, V).permute(0, 1, 3, 4, 2).contiguous()  # n,m,t,v,c  # noqa
        x = x.view(N, M*T, C*V)  # n,mt,vc

        if self.cls_token is not None:
            cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)

        x = self.trans_enc(x)
        if self.classifier_type == 'CLS':
            x = x[:, 0, :]  # n,vc
        elif self.classifier_type == 'GAP':
            x = x.mean(1)  # n,vc
        else:
            raise ValueError("Unknown classifier_type")

        return x, None

    def forward(self, x):
        size = x.size()
        x = self.forward_preprocess(x, size)
        if self.projection_layer:
            x = self.forward_model_backbone(x, size)
        x, _ = self.forward_postprocess(x, size)
        x = self.forward_classifier(x, size)
        return x


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph, model_layers=1,
                  trans_num_layers=10)
    # print(model)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model(torch.ones((1, 3, 300, 25, 2)))
