import torch
from torch import nn
from torchinfo import summary

import numpy as np
import math
from typing import Optional

from model.aagcn import import_class
from model.aagcn import TCNGCNUnit
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


class MHAUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_heads: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=True
        )
        self.norm = nn.LayerNorm(
            normalized_shape=in_channels,
            eps=1e-05,
            elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        # x : N, L, C
        x = self.norm(x)
        mha_x, mha_attn = self.mha(x, x, x)
        x = x + self.dropout(mha_x)
        return x, mha_attn


class FFNUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 inter_channels: int,
                 out_channels: int,
                 skip: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.skip = skip
        self.l1 = nn.Linear(in_channels, inter_channels)
        self.l2 = nn.Linear(inter_channels, out_channels)
        self.re = nn.GELU()
        self.n1 = nn.LayerNorm(normalized_shape=in_channels,
                               eps=1e-05,
                               elementwise_affine=True)
        self.d1 = nn.Dropout(p=dropout)
        self.d2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        # x : N, L, C
        if self.skip:
            x = x + self.d2(self.l2(self.d1(self.re(self.l1(self.n1(x))))))
        else:
            x = self.d2(self.l2(self.d1(self.re(self.l1(self.n1(x))))))
        return x


class TransformerUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 inter_channels: int,
                 num_heads: int = 1,
                 mha_dropout: float = 0.0,
                 ffn_dropout: float = 0.0):
        super().__init__()
        self.mha = MHAUnit(in_channels=in_channels,
                           num_heads=num_heads,
                           dropout=mha_dropout)
        self.ffn = FFNUnit(in_channels=in_channels,
                           inter_channels=inter_channels,
                           out_channels=in_channels,
                           skip=True,
                           dropout=ffn_dropout)

    def forward(self, x: torch.Tensor):
        # x : N, L, C
        mha_x, mha_attn = self.mha(x)
        ffn_x = self.ffn(x + mha_x)
        return ffn_x, mha_attn


class TransformerEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 inter_channels: int,
                 num_heads: int = 1,
                 num_layers: int = 1,
                 mha_dropout: float = 0.0,
                 ffn_dropout: float = 0.0,
                 pos_enc: bool = True,
                 classifier_type: str = 'CLS'):
        super().__init__()

        if pos_enc:
            self.pos_encoder = PositionalEncoding(in_channels)
        else:
            self.pos_encoder = lambda x: x

        self.classifier_type = classifier_type
        if classifier_type == 'CLS':
            self.cls_token = nn.Parameter(torch.randn(1, 1, in_channels))
        else:
            self.cls_token = None

        self.transformer_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_layers.append(
                TransformerUnit(in_channels=in_channels,
                                inter_channels=inter_channels,
                                num_heads=num_heads,
                                mha_dropout=mha_dropout,
                                ffn_dropout=ffn_dropout))

    def forward(self, x: torch.Tensor):
        # x : N, L, C
        if self.cls_token is not None:
            cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)
        ffn_x, mha_attn = [], []
        for layer in self.transformer_layers:
            ffn_x_l, mha_attn_l = layer(x)
            x = ffn_x_l
            ffn_x.append(ffn_x_l)
            mha_attn.append(mha_attn_l)
        if self.classifier_type == 'CLS':
            out = ffn_x[-1][:, 0, :]  # n,c
        elif self.classifier_type == 'GAP':
            out = ffn_x[-1].mean(1)  # n,c
        elif self.classifier_type == 'ALL':
            out = ffn_x[-1].view(ffn_x[-1].size(0), -1)  # n,tc
        else:
            raise ValueError("Unknown classifier_type")
        return out, ffn_x, mha_attn


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
# - uses transformer with feature projection.
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
                 trans_num_layers: int = 1,

                 num_heads: int = 1,
                 pos_enc: bool = True,
                 classifier_type: str = 'CLS',
                 attention_type: str = 'MT-VC',
                 attention_layers: int = 1,
                 mha_dropout: float = 0.0,
                 ffn_dropout: float = 0.0,
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

        # if graph is None:
        #     raise ValueError()
        # else:
        #     Graph = import_class(graph)
        #     self.graph = Graph(**graph_args)

        # def _TCNGCNUnit(_in, _out, stride=1, residual=True):
        #     return TCNGCNUnit(_in,
        #                       _out,
        #                       self.graph.A,
        #                       num_subset=num_subset,
        #                       stride=stride,
        #                       residual=residual,
        #                       adaptive=self.adaptive_fn,
        #                       attention=attention,
        #                       gbn_split=gbn_split)

        # self.init_model_backbone(model_layers=model_layers,
        #                          tcngcn_unit=_TCNGCNUnit)

        # self.attention_type = attention_type
        # if self.attention_type == 'T-MVC':
        #     self.proj = FFNUnit(in_channels=256*num_point*num_person,
        #                         inter_channels=(256*num_point*num_person)//8,
        #                         out_channels=256,
        #                         skip=False)
        # elif self.attention_type == 'MT-VC':
        #     self.proj = FFNUnit(in_channels=256*num_point,
        #                         inter_channels=(256*num_point)//8,
        #                         out_channels=256,
        #                         skip=False)
        # elif self.attention_type == 'T-VC':
        #     self.proj = FFNUnit(in_channels=256*num_point,
        #                         inter_channels=(256*num_point)//8,
        #                         out_channels=256,
        #                         skip=False)
        # else:
        #     raise ValueError("Unknown attention_type")

        # self.trans = TransformerEncoder(
        #     in_channels=256,
        #     inter_channels=256*4,
        #     num_heads=num_heads,
        #     num_layers=attention_layers,
        #     mha_dropout=mha_dropout,
        #     ffn_dropout=ffn_dropout,
        #     pos_enc=pos_enc,
        #     classifier_type=classifier_type,
        # )

        # if classifier_type == 'ALL':
        #     self.init_fc(256*75, num_class)
        # else:
        #     self.init_fc(256, num_class)

    # def forward_postprocess(self, x, size):
    #     # x : NM C T V
    #     N, _, _, V, M = size
    #     _, C, T, _ = x.size()
    #     x = x.view(N, M, C, T, V)   # n,m,c,t,v
    #     if self.attention_type == 'T-MVC':
    #         x = x.permute(0, 3, 1, 4, 2).contiguous()   # n,t,m,v,c
    #         x = x.view(N, T, M*C*V)   # n,t,mvc
    #         x = self.proj(x)   # n,t,mvc
    #         x, x_list, a = self.trans(x)   # n,c
    #     elif self.attention_type == 'MT-VC':
    #         x = x.permute(0, 1, 3, 4, 2).contiguous()   # n,m,t,v,c
    #         x = x.view(N, M*T, C*V)   # n,mt,vc
    #         x = self.proj(x)   # n,mt,c
    #         x, x_list, a = self.trans(x)   # n,c
    #     elif self.attention_type == 'T-VC':
    #         x = x.permute(0, 1, 3, 4, 2).contiguous()   # n,m,t,v,c
    #         x = x.view(N*M, T, C*V)   # nm,t,vc
    #         x = self.proj(x)   # nm,t,c
    #         x, x_list, a = self.trans(x)   # nm,c
    #         x = x.view(N, M, -1).mean(1)   # n,c
    #     else:
    #         raise ValueError("Unknown attention_type")
    #     return x, a


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph, model_layers=1,
                  trans_num_layers=10, num_heads=2,)
    # print(model)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model(torch.ones((1, 3, 300, 25, 2)))
