import torch
from torch import nn
from torchinfo import summary

import copy
import numpy as np
import math
from typing import Optional

from model.architecture.aagcn.aagcn import TCNGCNUnit
from model.architecture.aagcn.aagcn import BaseModel


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
            src1, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src1)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # noqa
            src = src + self.dropout2(src2)
            return src, attn

        else:
            src1, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src1)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # noqa
            src = src + self.dropout2(src2)
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
# - from v17
# - 1 PE
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

                 backbone_dim: int = 16,
                 t_trans_cfg: Optional[dict] = None,
                 s_trans_cfg: Optional[dict] = None,

                 add_A: bool = False,
                 add_Aa: bool = False,

                 trans_seq: str = 's-t',

                 multi_trans_dropout: float = 0.0,
                 res_dropout: float = 0.2,

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

        self.trans_seq = trans_seq

        self.num_subset = num_subset
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
                                 output_channel=backbone_dim)

        # 3. transformer (temporal)
        t_trans_dim = t_trans_cfg['model_dim'] * num_point
        t_trans_cfg['model_dim'] = t_trans_dim
        t_trans_cfg['ffn_dim'] *= num_point
        t_trans_enc_layer = TransformerEncoderLayerExtV2(cfg=t_trans_cfg)
        self.t_trans_enc_layers = torch.nn.ModuleList(
            [copy.deepcopy(t_trans_enc_layer)
             for _ in range(t_trans_cfg['num_layers'])])

        pos_enc = str(pos_enc)
        if pos_enc == 'True' or pos_enc == 'original':
            self.t_pos_encoder = PositionalEncoding(t_trans_dim)
        elif pos_enc == 'cossin':
            self.t_pos_encoder = CosSinPositionalEncoding(t_trans_dim)
        else:
            self.t_pos_encoder = lambda x: x

        if 'res' in trans_seq:
            self.res_dropout = nn.Dropout(p=res_dropout)
            self.res_norm = nn.LayerNorm(t_trans_dim,
                                         eps=t_trans_cfg['layer_norm_eps'])

        # 4. transformer (spatial)
        self.alpha = 1
        self.sa_norm = lambda x: x
        self.multi_trans_dropout = nn.Dropout(p=multi_trans_dropout)
        if add_A:
            s_trans_dim = s_trans_cfg['model_dim'] * 100
            s_trans_cfg['model_dim'] = s_trans_dim
            s_trans_cfg['ffn_dim'] *= 100
            s_trans_enc_layers = torch.nn.ModuleDict(
                {
                    f'subset{a_i}':
                    copy.deepcopy(
                        TransformerEncoderLayerExtV2(
                            cfg=s_trans_cfg,
                            A=self.graph.A[a_i]
                        )
                    )
                    for a_i in range(self.num_subset)
                }
            )
            self.s_trans_enc_layers = torch.nn.ModuleList(
                [copy.deepcopy(s_trans_enc_layers)
                 for _ in range(s_trans_cfg['num_layers'])]
            )
            self.sa_norm = nn.LayerNorm(s_trans_dim,
                                        eps=s_trans_cfg['layer_norm_eps'])
            if add_Aa:
                self.alpha = nn.Parameter(torch.zeros(1))
        else:
            s_trans_dim = s_trans_cfg['model_dim'] * 100
            s_trans_cfg['model_dim'] = s_trans_dim
            s_trans_cfg['ffn_dim'] *= 100
            s_trans_enc_layer = TransformerEncoderLayerExtV2(
                cfg=s_trans_cfg,
                A=None
            )
            self.s_trans_enc_layers = torch.nn.ModuleList(
                [copy.deepcopy(s_trans_enc_layer)
                 for _ in range(s_trans_cfg['num_layers'])]
            )

        # 5. classifier
        self.classifier_type = classifier_type
        if classifier_type == 'CLS':
            self.cls_token = nn.Parameter(torch.randn(1, 1, t_trans_dim))
        else:
            self.cls_token = None

        self.init_fc(t_trans_dim, num_class)

    def forward_postprocess(self, x: torch.Tensor, size: torch.Size):
        N, _, _, V, M = size
        _, C, T, _ = x.size()
        x = x.view(N, M, C, T, V).permute(0, 1, 3, 4, 2).contiguous()  # n,m,t,v,c  # noqa

        x = x.reshape(N, M*T, V*C)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.t_pos_encoder(x)

        attn = [[], []]

        for i in range(len(self.t_trans_enc_layers)):

            if isinstance(self.s_trans_enc_layers[i], torch.nn.ModuleDict):
                s_trans_enc_layers = self.s_trans_enc_layers[i].items()
            else:
                s_trans_enc_layers = [(None, self.s_trans_enc_layers[i])]

            if self.trans_seq == 's-t':
                # Spatial
                x0 = x[:, 0:1, :]  # n,1,vc
                x = x[:, 1:, :]  # n,mt,vc
                x = x.view(N, M, T, V, C).permute(0, 1, 3, 2, 4).contiguous()
                x = x.reshape(N, M*V, T*C)  # n,mv,tc
                x, a = self.s_trans_enc_layers[i](x)
                if self.need_attn:
                    attn[1].append(a)
                x = x.view(N, M, V, T, C).permute(0, 1, 3, 2, 4).contiguous()
                x = x.reshape(N, M*T, V*C)  # n,mv,tc
                x = torch.cat([x0, x], dim=1)

                # Temporal
                x = x.reshape(N, -1, V*C)  # n,mt,vc
                x, a = self.t_trans_enc_layers[i](x)
                if self.need_attn:
                    attn[0].append(a)

                # x = x.reshape(-1, V, C)  # nmt,v,c
                # A = s_layer.PA  # 3,v,v
                # mask = None if A is None else A.repeat(N*(M*T+1), 1, 1)
                # x, a = s_layer(x, mask)
                # if self.need_attn:
                #     attn[1].append(a)

            elif self.trans_seq == 'sa-t':
                # Spatial
                x0 = x[:, 0:1, :]  # n,1,vc
                x = x[:, 1:, :]  # n,mt,vc
                x = x.view(N, M, T, V, C).permute(0, 1, 3, 2, 4).contiguous()
                x = x.reshape(N*M, V, T*C)  # nm,v,tc

                x_l = []
                for _, s_layer in s_trans_enc_layers:
                    # v,v
                    if s_layer.PA is None:
                        mask = None
                    else:
                        mask = s_layer.PA * self.alpha
                    x_i, a = s_layer(x, mask)
                    x_l.append(x_i)
                    if self.need_attn:
                        attn[1].append(a)
                x = x + torch.stack(x_l, dim=0).sum(dim=0)
                x = self.sa_norm(x)

                x = x.view(N, M, V, T, C).permute(0, 1, 3, 2, 4).contiguous()
                x = x.reshape(N, M*T, V*C)  # n,mv,tc
                x = torch.cat([x0, x], dim=1)

                # Temporal
                x = x.reshape(N, -1, V*C)  # n,mt,vc
                x, a = self.t_trans_enc_layers[i](x)
                if self.need_attn:
                    attn[0].append(a)

            elif self.trans_seq == 'sa-t-res':
                # Spatial
                x0 = x[:, 0:1, :]  # n,1,vc
                x1 = x[:, 1:, :]  # n,mt,vc
                x1 = x1.view(N, M, T, V, C).permute(0, 1, 3, 2, 4).contiguous()
                x1 = x1.reshape(N*M, V, T*C)  # nm,v,tc

                x_l = []
                for _, s_layer in s_trans_enc_layers:
                    # v,v
                    if s_layer.PA is None:
                        mask = None
                    else:
                        mask = s_layer.PA * self.alpha
                    x_i, a = s_layer(x1, mask)
                    x_l.append(x_i)
                    if self.need_attn:
                        attn[1].append(a)
                x_l = torch.stack(x_l, dim=0).sum(dim=0)
                x1 = x1 + self.multi_trans_dropout(x_l)
                x1 = self.sa_norm(x1)

                x1 = x1.view(N, M, V, T, C).permute(0, 1, 3, 2, 4).contiguous()
                x1 = x1.reshape(N, M*T, V*C)  # n,mv,tc
                x2 = torch.cat([x0, x1], dim=1)

                # Temporal
                x2 = x2.reshape(N, -1, V*C)  # n,mt,vc
                x2, a = self.t_trans_enc_layers[i](x2)
                if self.need_attn:
                    attn[0].append(a)

                # Residual
                x = x + self.res_dropout(x2)
                x = self.res_norm(x)

            elif self.trans_seq == 't-s':
                # Temporal
                x = x.reshape(N, -1, V*C)  # n,mt,vc
                x, a = self.t_trans_enc_layers[i](x)
                if self.need_attn:
                    attn[0].append(a)

                # Spatial
                x0 = x[:, 0: 1, :]  # n,1,vc
                x = x[:, 1:, :]  # n,mt,vc
                x = x.view(N, M, T, V, C).permute(0, 1, 3, 2, 4).contiguous()
                x = x.reshape(N, M*V, T*C)  # n,mv,tc
                x, a = self.s_trans_enc_layers[i](x)
                if self.need_attn:
                    attn[1].append(a)
                x = x.view(N, M, V, T, C).permute(0, 1, 3, 2, 4).contiguous()
                x = x.reshape(N, M*T, V*C)  # n,mv,tc
                x = torch.cat([x0, x], dim=1)

        x = x.reshape(N, -1, V*C)  # n,mt,vc
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
                  trans_seq='sa-t-res',
                  add_A=False,
                  kernel_size=3,
                  pad=False,
                  pos_enc='cossin'
                  )
    # print(model)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print([i[0] for i in model.named_parameters() if 'PA' in i[0]])
    model(torch.ones((3, 3, 300, 25, 2)))
