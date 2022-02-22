import torch
from torch import nn

import copy
import numpy as np
import math
from typing import Optional

from model.aagcn import conv_init
from model.aagcn import bn_init
from model.aagcn import batch_norm_2d
from model.aagcn import GCNUnit
from model.aagcn import AdaptiveGCN
from model.aagcn import BaseModel

from model.multiheadattention import MultiheadAttention


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
                 mha: nn.Module,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 device=None,
                 dtype=None,
                 pre_norm: bool = False,
                 pos_emb: dict = None,
                 A: np.ndarray = None,
                 Aa: bool = False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, device, dtype)
        self.pre_norm = pre_norm
        if A is None:
            self.PA = None
        else:
            self.PA = nn.Parameter(
                torch.from_numpy(A.astype(np.float32)))  # Bk
        if Aa is None:
            self.alpha = None
        else:
            self.alpha = nn.Parameter(torch.zeros(1))
        if mha == MultiheadAttention:
            kwargs = {
                'embed_dim': d_model,
                'num_heads': nhead,
                'dropout': dropout,
                'batch_first': batch_first,
            }
            kwargs.update(factory_kwargs)
            kwargs['pos_emb'] = pos_emb
            self.self_attn = MultiheadAttention(**kwargs)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                alpha: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        kwargs = {
            'attn_mask': src_mask,
            'key_padding_mask': src_key_padding_mask,
        }
        if self.self_attn == MultiheadAttention:
            kwargs['alpha'] = alpha

        if self.pre_norm:
            src = self.norm1(src)
            src1, attn = self.self_attn(src, src, src, **kwargs)
            src = src + self.dropout1(src1)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # noqa
            src = src + self.dropout2(src2)
            return src, attn

        else:
            src1, attn = self.self_attn(src, src, src, **kwargs)
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
                 mha: nn.Module,
                 A: np.ndarray = None,
                 Aa: bool = False) -> None:
        pos_emb = {
            'name': cfg['pos_emb'],
            'tokens': cfg['length'],
            'dim_head': cfg['model_dim']//cfg['num_heads'],
            'heads': True if 'share' in cfg['pos_emb'] else False,
        }
        super().__init__(
            mha=mha,
            d_model=cfg['model_dim'],
            nhead=cfg['num_heads'],
            dim_feedforward=cfg['ffn_dim'],
            dropout=cfg['dropout'],
            activation=cfg['activation'],
            layer_norm_eps=cfg['layer_norm_eps'],
            batch_first=cfg['batch_first'],
            pre_norm=cfg['prenorm'],
            pos_emb=pos_emb,
            A=A,
            Aa=Aa,
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
        'num_layers',
        'length',
        'pos_emb'
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
                 trans_len: int = 100,

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

        if t_trans_cfg is not None:
            t_trans_cfg['layer_norm_eps'] = 1e-5
            t_trans_cfg['batch_first'] = True
            transformer_config_checker(t_trans_cfg)
        if s_trans_cfg is not None:
            s_trans_cfg['layer_norm_eps'] = 1e-5
            s_trans_cfg['batch_first'] = True
            transformer_config_checker(s_trans_cfg)

        self.trans_seq = trans_seq
        if 'v2' in self.trans_seq:
            mha = MultiheadAttention
        else:
            mha = nn.MultiheadAttention

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
        if t_trans_cfg is not None:
            t_trans_dim = t_trans_cfg['model_dim'] * num_point
            t_trans_cfg['model_dim'] = t_trans_dim
            t_trans_cfg['ffn_dim'] *= num_point
            t_trans_enc_layer = TransformerEncoderLayerExtV2(
                cfg=t_trans_cfg,
                mha=mha
            )
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
        if s_trans_cfg is not None:
            self.sa_norm = lambda x: x
            self.multi_trans_dropout = nn.Dropout(p=multi_trans_dropout)
            s_trans_dim = s_trans_cfg['model_dim'] * trans_len
            s_trans_cfg['model_dim'] = s_trans_dim
            s_trans_cfg['ffn_dim'] *= trans_len
            if add_A:
                s_trans_enc_layers = torch.nn.ModuleDict(
                    {
                        f'subset{a_i}':
                        copy.deepcopy(
                            TransformerEncoderLayerExtV2(
                                cfg=s_trans_cfg,
                                mha=mha,
                                A=self.graph.A[a_i],
                                Aa=add_Aa
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
            else:
                s_trans_enc_layer = TransformerEncoderLayerExtV2(
                    cfg=s_trans_cfg,
                    mha=mha,
                    A=None
                )
                self.s_trans_enc_layers = torch.nn.ModuleList(
                    [copy.deepcopy(s_trans_enc_layer)
                     for _ in range(s_trans_cfg['num_layers'])]
                )

        # 5. classifier
        self.classifier_type = classifier_type
        if 'CLS' in classifier_type:
            self.cls_token = nn.Parameter(torch.randn(1, 1, t_trans_dim))
        else:
            self.cls_token = None
        if 'POOL' in classifier_type:
            self.cls_pool_fc = nn.Linear(t_trans_dim, t_trans_dim)
            self.cls_pool_act = nn.Tanh()

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
                        mask = s_layer.PA * s_layer.alpha
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
                        mask = s_layer.PA * s_layer.alpha
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

            elif self.trans_seq == 's-t-v2' or \
                    self.trans_seq == 's-t-res-v2':
                # Spatial
                x0 = x[:, 0:1, :]  # n,1,vc
                x1 = x[:, 1:, :]  # n,mt,vc
                x1 = x1.view(N, M, T, V, C).permute(0, 1, 3, 2, 4).contiguous()
                x1 = x1.reshape(N, M*V, T*C)  # n,mv,tc
                x1, a = self.s_trans_enc_layers[i](x1)
                if self.need_attn:
                    attn[1].append(a)
                x1 = x1.view(N, M, V, T, C).permute(0, 1, 3, 2, 4).contiguous()
                x1 = x1.reshape(N, M*T, V*C)  # n,mv,tc
                x2 = torch.cat([x0, x1], dim=1)

                # Temporal
                x2 = x2.reshape(N, -1, V*C)  # n,mt,vc
                x2, a = self.t_trans_enc_layers[i](x2)
                if self.need_attn:
                    attn[0].append(a)

                if 'res' in self.trans_seq:
                    # Residual
                    x = x + self.res_dropout(x2)
                    x = self.res_norm(x)
                else:
                    x = x2

            elif self.trans_seq == 'sa-t-v2' or \
                    self.trans_seq == 'sa-t-res-v2':
                # Spatial
                x0 = x[:, 0:1, :]  # n,1,vc
                x1 = x[:, 1:, :]  # n,mt,vc
                x1 = x1.view(N, M, T, V, C).permute(0, 1, 3, 2, 4).contiguous()
                x1 = x1.reshape(N*M, V, T*C)  # nm,v,tc

                x_l = []
                for _, s_layer in s_trans_enc_layers:
                    # v,v
                    mask = s_layer.PA
                    x_i, a = s_layer(x1, mask, alpha=s_layer.alpha)
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

                if 'res' in self.trans_seq:
                    # Residual
                    x = x + self.res_dropout(x2)
                    x = self.res_norm(x)
                else:
                    x = x2

        x = x.reshape(N, -1, V*C)  # n,mt,vc
        if self.classifier_type == 'CLS':
            x = x[:, 0, :]  # n,vc
        elif self.classifier_type == 'CLS-POOL':
            x = x[:, 0, :]  # n,vc
            x = self.cls_pool_fc(x)
            x = self.cls_pool_act(x)
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
                      'num_layers': 2,
                      'pos_emb': 'rel-shared',
                      'length': 201,
                  },
                  s_trans_cfg={
                      'num_heads': 2,
                      'model_dim': 16,
                      'ffn_dim': 64,
                      'dropout': 0,
                      'activation': 'gelu',
                      'prenorm': False,
                      'num_layers': 2,
                      'pos_emb': 'rel-shared',
                      'length': 25,
                  },
                  trans_seq='sa-t-v2',
                  add_A=True,
                  add_Aa=True,
                  kernel_size=3,
                  pad=False,
                  pos_enc=None,
                  classifier_type='CLS-POOL'
                  )
    # print(model)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print([i[0] for i in model.named_parameters() if 'PA' in i[0]])
    model(torch.ones((3, 3, 300, 25, 2)))