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


class TransformerEncoderLayerExtV2(TransformerEncoderLayerExt):
    """Option for pre of postnorm"""

    def __init__(self,
                 cfg: dict,
                 A: np.ndarray = None) -> None:
        super().__init__(
            d_model=cfg['model_dim'],
            nhead=cfg['num_heads'],
            dim_feedforward=cfg['ffn_dim'],
            dropout=cfg['dropout'],
            activation=cfg['activation'],
            layer_norm_eps=cfg['layer_norm_eps'],
            batch_first=cfg['batch_first'],
            pre_norm=cfg['prenorm'],
            A=A,
        )


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


def transformer_config_checker(cfg: dict):
    trans_cfg_names = [
        'num_heads',
        'model_dim',
        'ffn_dim',
        'dropout',
        'activation',
        'prenorm',
        'batch_first',
        'layer_norm_eps',
        'num_layers'
    ]
    for x in cfg:
        assert f'{x}' in trans_cfg_names, f'{x} not in transformer config'


# ------------------------------------------------------------------------------
# Network
# - uses original transformer
# - from v20
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
                 t_trans_cfg: Optional[dict] = None,
                 s_trans_cfg: Optional[dict] = None,
                 add_A: bool = False,
                 pos_enc: str = 'True',
                 classifier_type: str = 'CLS',
                 model_layers: int = 10
                 ):

        super().__init__(num_class, num_point, num_person,
                         in_channels, drop_out, adaptive, gbn_split)

        t_trans_cfg['layer_norm_eps'] = 1e-5
        s_trans_cfg['layer_norm_eps'] = 1e-5
        t_trans_cfg['batch_first'] = True
        s_trans_cfg['batch_first'] = True
        transformer_config_checker(t_trans_cfg)
        transformer_config_checker(s_trans_cfg)

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
                                 output_channel=t_trans_cfg['model_dim'])

        # 3. transformer (temporal)
        t_trans_dim = t_trans_cfg['model_dim'] * num_point
        t_trans_cfg['model_dim'] = t_trans_dim
        t_trans_enc_layer = TransformerEncoderLayerExtV2(cfg=t_trans_cfg)
        self.t_trans_enc_layers = torch.nn.ModuleList(
            [copy.deepcopy(t_trans_enc_layer)
             for _ in range(t_trans_cfg['num_layers'])])

        pos_enc = str(pos_enc)
        if pos_enc == 'True' or pos_enc == 'original':
            self.t_pos_encoder = nn.ModuleList([
                PositionalEncoding(t_trans_dim)
                for _ in range(t_trans_cfg['num_layers'])
            ])
        elif pos_enc == 'cossin':
            self.t_pos_encoder = nn.ModuleList([
                CosSinPositionalEncoding(t_trans_dim, name=str(i))
                for i in range(t_trans_cfg['num_layers'])
            ])
        else:
            self.t_pos_encoder = [lambda x: x
                                  for _ in range(t_trans_cfg['num_layers'])]

        # 4. transformer (spatial)
        s_trans_dim = s_trans_cfg['model_dim']
        s_trans_enc_layer = TransformerEncoderLayerExtV2(
            cfg=s_trans_cfg,
            A=self.graph.A if add_A else None
        )
        self.s_trans_enc_layers = torch.nn.ModuleList(
            [copy.deepcopy(s_trans_enc_layer)
             for _ in range(s_trans_cfg['num_layers'])])

        pos_enc = str(pos_enc)
        if pos_enc == 'True' or pos_enc == 'original':
            self.s_pos_encoder = nn.ModuleList([
                PositionalEncoding(s_trans_dim, max_len=100)
                for _ in range(s_trans_cfg['num_layers'])
            ])
        elif pos_enc == 'cossin':
            self.s_pos_encoder = nn.ModuleList([
                CosSinPositionalEncoding(s_trans_dim, max_len=100, name=str(i))
                for i in range(s_trans_cfg['num_layers'])
            ])
        else:
            self.s_pos_encoder = [lambda x: x
                                  for _ in range(s_trans_cfg['num_layers'])]

        # 5. classifier
        self.classifier_type = classifier_type
        if classifier_type == 'CLS':
            self.s_cls_token = nn.Parameter(torch.randn(1, 1, s_trans_dim))
            self.t_cls_token = nn.Parameter(torch.randn(1, 1, t_trans_dim))
        else:
            self.s_cls_token = None
            self.t_cls_token = None

        self.init_fc(t_trans_dim//num_point * (1+num_point), num_class)

    def forward_postprocess(self, x: torch.Tensor, size: torch.Size):
        N, _, _, V, M = size
        _, C, T, _ = x.size()

        s_x = x.view(N, M, C, T, V).permute(0, 3, 1, 4, 2).contiguous()  # n,t,m,v,c  # noqa
        s_x = s_x.reshape(N*T, M*V, C)  # nt,mv,c
        if self.s_cls_token is not None:
            cls_tokens = self.s_cls_token.repeat(N*T, 1, 1)
            s_x = torch.cat((cls_tokens, s_x), dim=1)
        # s_x = self.s_pos_encoder(s_x)  # nt,mv+1,c

        t_x = x.view(N, M, C, T, V).permute(0, 1, 3, 4, 2).contiguous()  # n,m,t,v,c  # noqa
        t_x = t_x.reshape(N, M*T, V*C)  # n,mt,vc
        if self.t_cls_token is not None:
            cls_tokens = self.t_cls_token.repeat(N, 1, 1)
            t_x = torch.cat((cls_tokens, t_x), dim=1)
        # t_x = self.t_pos_encoder(t_x)  # n,mt+1,vc

        attn = [[], []]
        for s_layer, t_layer, s_pe, t_pe in zip(self.s_trans_enc_layers,
                                                self.t_trans_enc_layers,
                                                self.s_pos_encoder,
                                                self.t_pos_encoder):
            s_x = s_pe(s_x)
            s_x, a = s_layer(s_x)
            if self.need_attn:
                attn[0].append(a)
            t_x = t_pe(t_x)
            t_x, a = t_layer(t_x)
            if self.need_attn:
                attn[1].append(a)

        if self.classifier_type == 'CLS':
            s_x = s_x[:, 0, :]  # nt,c
            s_x = s_x.reshape(N, -1, C)  # n,t,c
            s_x = s_x.mean(1)
            t_x = t_x[:, 0, :]  # n,vc
            x = torch.cat([s_x, t_x], dim=1)  # n,(v+1)c
        # elif self.classifier_type == 'GAP':
        #     x = x.mean(1)  # n,vc
        else:
            raise ValueError("Unknown classifier_type")

        return x, attn


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph,
                  model_layers=101,
                  t_trans_cfg={
                      'num_heads': 2,
                      'model_dim': 16,
                      'ffn_dim': 64,
                      'dropout': 0,
                      'activation': 'gelu',
                      'prenorm': False,
                      'num_layers': 3
                  },
                  s_trans_cfg={
                      'num_heads': 2,
                      'model_dim': 16,
                      'ffn_dim': 64,
                      'dropout': 0,
                      'activation': 'gelu',
                      'prenorm': False,
                      'num_layers': 3
                  },
                  #   trans_num_layers=3,
                  #   trans_model_dim=24,
                  #   trans_ffn_dim=96,
                  #   trans_num_heads=3,
                  #   s_trans_num_layers=3,
                  #   s_trans_model_dim=24,
                  #   s_trans_ffn_dim=96,
                  #   s_trans_num_heads=3,
                  #   add_A=True,
                  kernel_size=3,
                  pad=False,
                  pos_enc='cossin'
                  )
    # print(model)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print([i[0] for i in model.named_parameters() if 'PA' in i[0]])
    model(torch.ones((3, 3, 300, 25, 2)))
