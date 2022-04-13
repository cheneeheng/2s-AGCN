import torch
from torch import nn

import copy
import numpy as np
import math
from typing import Optional

from model.model.aagcn.aagcn import TCNGCNUnit
from model.model.aagcn.aagcn import BaseModel

from model.module.attention.multiheadattention import MultiheadAttention


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
                 Aa: str = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, device, dtype)
        self.pre_norm = pre_norm
        if A is None:
            self.PA = None
        else:
            self.PA = nn.Parameter(
                torch.from_numpy(A.astype(np.float32)))  # Bk
        if Aa == 'None' or Aa == 'False':
            self.alpha = None
        elif Aa == 'True' or Aa == 'zero':
            self.alpha = nn.Parameter(torch.zeros(1))
        elif Aa == 'one':
            self.alpha = nn.Parameter(torch.ones(1))
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
                alpha: Optional[torch.Tensor] = None,
                global_attn: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        kwargs = {
            'attn_mask': src_mask,
            'key_padding_mask': src_key_padding_mask,
        }
        if isinstance(self.self_attn, MultiheadAttention):
            kwargs['alpha'] = alpha
            kwargs['global_attn'] = global_attn

        if self.pre_norm:
            src = self.norm1(src)
            output = self.self_attn(src, src, src, **kwargs)
            src1 = output[0]
            src = src + self.dropout1(src1)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # noqa
            src = src + self.dropout2(src2)
            return src, *output[1:]

        else:
            output = self.self_attn(src, src, src, **kwargs)
            src1 = output[0]
            src = src + self.dropout1(src1)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # noqa
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src, *output[1:]


class TransformerEncoderLayerExtV2(TransformerEncoderLayerExt):
    """Option for pre of postnorm"""

    def __init__(self,
                 cfg: dict,
                 mha: nn.Module,
                 A: np.ndarray = None,
                 Aa: str = None) -> None:
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
                 add_Aa: str = None,

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
        if 'v2' in self.trans_seq or 'v3' in self.trans_seq:
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
                                Aa=str(add_Aa)
                            )
                        )
                        for a_i in range(self.num_subset)
                    }
                )
                if 'v3' in self.trans_seq:
                    s_trans_enc_layers.update(
                        {
                            'sa_norm': nn.LayerNorm(
                                s_trans_dim, eps=s_trans_cfg['layer_norm_eps']),
                            # 'sa_fc': nn.Linear(s_trans_dim, s_trans_dim)
                        }
                    )
                    self.s_trans_enc_layers = torch.nn.ModuleList(
                        [copy.deepcopy(s_trans_enc_layers)
                         for _ in range(s_trans_cfg['num_layers'])]
                    )
                else:
                    self.s_trans_enc_layers = torch.nn.ModuleList(
                        [copy.deepcopy(s_trans_enc_layers)
                         for _ in range(s_trans_cfg['num_layers'])]
                    )
                    self.sa_norm = nn.LayerNorm(
                        s_trans_dim, eps=s_trans_cfg['layer_norm_eps'])
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

        x = x.reshape(N*M, T, V*C)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.t_pos_encoder(x)

        def _temporal_trans(_layer, _x, _attn, _pe=False):
            out = _layer(_x)  # nm,1+t,vc
            _x, _a = out[0], out[1]
            if self.need_attn:
                _attn.append((_a, out[2]) if _pe else _a)
            return _x, _attn

        def _spatial_trans(_layer, _x, _attn, _pe=False):
            if self.cls_token is not None:
                _x0 = _x[:, 0:1, :]  # nm,1,vc
                _x = _x[:, 1:, :]  # nm,t,vc
            _x = _x.view(N, M, T, V, C).permute(0, 1, 3, 2, 4).contiguous()
            _x = _x.reshape(N*M, V, T*C)  # nm,v,tc
            out = _layer(_x)
            _x, _a = out[0], out[1]
            if self.need_attn:
                _attn.append((_a, out[2]) if _pe else _a)
            _x = _x.view(N, M, V, T, C).permute(0, 1, 3, 2, 4).contiguous()
            _x = _x.reshape(N*M, T, V*C)  # nm,t,vc
            if self.cls_token is not None:
                _x = torch.cat([_x0, _x], dim=1)  # nm,1+t,vc
            return _x, _attn

        def _spatial_Aa_trans(_layers, _x, _attn, _pe=False, mode=None):
            if self.cls_token is not None:
                _x0 = _x[:, 0:1, :]  # nm,1,vc
                _x1 = _x[:, 1:, :]  # nm,t,vc
            else:
                _x1 = _x[:, :, :]  # nm,t,vc
            _x1 = _x1.view(N, M, T, V, C).permute(0, 1, 3, 2, 4).contiguous()
            _x1 = _x1.reshape(N*M, V, T*C)  # nm,v,tc
            _x_l = []
            if mode == 'v3':
                for _, s_layer in list(_layers)[:-1]:
                    # v,v
                    mask = s_layer.PA
                    alpha = s_layer.alpha
                    out = s_layer(_x1, global_attn=mask, alpha=alpha)
                    _x_l.append(out[0])
                    if self.need_attn:
                        _attn.append((out[1], out[2]) if _pe else out[1])
            else:
                for _, s_layer in _layers:
                    # v,v
                    if s_layer.PA is None:
                        mask = None
                    else:
                        mask = s_layer.PA * s_layer.alpha
                    out = s_layer(_x1, global_attn=mask)  # x, a, pe
                    _x_l.append(out[0])
                    if self.need_attn:
                        _attn.append((out[1], out[2]) if _pe else out[1])
            _x_l = torch.stack(_x_l, dim=0).sum(dim=0)
            if mode == 'v3':
                _x1 = self.multi_trans_dropout(_x_l)
                _x1 = list(_layers)[-1][1](_x1)
            else:
                _x1 = _x1 + self.multi_trans_dropout(_x_l)
                _x1 = self.sa_norm(_x1)
            _x1 = _x1.view(N, M, V, T, C).permute(0, 1, 3, 2, 4).contiguous()
            _x1 = _x1.reshape(N*M, T, V*C)  # nm,t,vc
            if self.cls_token is not None:
                _x1 = torch.cat([_x0, _x1], dim=1)  # nm,1+t,vc
            return _x1, _attn

        attn = [[], []]
        for i in range(len(self.t_trans_enc_layers)):

            if isinstance(self.s_trans_enc_layers[i], torch.nn.ModuleDict):
                s_trans_enc_layers = self.s_trans_enc_layers[i].items()
            else:
                s_trans_enc_layers = [(None, self.s_trans_enc_layers[i])]

            if self.trans_seq == 't-s':
                # Temporal
                x1, attn[0] = _temporal_trans(
                    self.t_trans_enc_layers[i], x, attn[0])
                # Spatial
                x2, attn[1] = _spatial_trans(
                    self.s_trans_enc_layers[i], x1, attn[1])

            elif self.trans_seq == 's-t':
                # Spatial
                x1, attn[0] = _spatial_trans(
                    self.s_trans_enc_layers[i], x, attn[0])
                # Temporal
                x2, attn[1] = _temporal_trans(
                    self.t_trans_enc_layers[i], x1, attn[1])

            elif self.trans_seq == 'sa-t' or self.trans_seq == 'sa-t-res':
                # Spatial
                x1, attn[0] = _spatial_Aa_trans(
                    s_trans_enc_layers, x, attn[0])
                # Temporal
                x2, attn[1] = _temporal_trans(
                    self.t_trans_enc_layers[i], x1, attn[1])

            elif self.trans_seq == 's-t-v2' or \
                    self.trans_seq == 's-t-res-v2':
                # Spatial
                x1, attn[0] = _spatial_trans(
                    self.s_trans_enc_layers[i], x, attn[0], _pe=True)
                # Temporal
                x2, attn[1] = _temporal_trans(
                    self.t_trans_enc_layers[i], x1, attn[1], _pe=True)

            elif self.trans_seq == 'sa-t-v2' or \
                    self.trans_seq == 'sa-t-res-v2':
                # Spatial
                x1, attn[0] = _spatial_Aa_trans(
                    s_trans_enc_layers, x, attn[0], _pe=True)
                # Temporal
                x2, attn[1] = _temporal_trans(
                    self.t_trans_enc_layers[i], x1, attn[1], _pe=True)

            elif self.trans_seq == 'sa-t-v3' or \
                    self.trans_seq == 'sa-t-res-v3':
                # Spatial
                x1, attn[0] = _spatial_Aa_trans(
                    s_trans_enc_layers, x, attn[0], _pe=True, mode='v3')
                # Temporal
                x2, attn[1] = _temporal_trans(
                    self.t_trans_enc_layers[i], x1, attn[1], _pe=True)

            # Residual
            if 'res' in self.trans_seq:
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
                      'num_layers': 3,
                      'pos_emb': 'rel-shared',
                      'length': 101,
                  },
                  s_trans_cfg={
                      'num_heads': 2,
                      'model_dim': 16,
                      'ffn_dim': 64,
                      'dropout': 0,
                      'activation': 'gelu',
                      'prenorm': False,
                      'num_layers': 3,
                      'pos_emb': 'rel-shared',
                      'length': 25,
                  },
                  trans_seq='sa-t-v3',
                  add_A=True,
                  add_Aa='one',
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
