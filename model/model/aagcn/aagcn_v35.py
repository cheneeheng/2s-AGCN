import torch
from torch import nn
from torch.nn.functional import gelu

import copy
import numpy as np
import math
from typing import Optional, Union

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
        Aa = str(Aa)
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

                 add_A: Optional[str] = None,
                 add_Aa: str = None,
                 invert_A: bool = False,

                 trans_seq: str = 's-t',

                 add_s_cls: bool = False,

                 m_mask: bool = False,

                 sa_dropout: float = 0.0,
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
        mha = MultiheadAttention

        self.num_subset = num_subset
        self.need_attn = need_attn

        self.m_mask = m_mask
        self.m_b_mask = None

        self.rel_emb_s = True if 'rel' in s_trans_cfg['pos_emb'] else False
        self.rel_emb_t = True if 'rel' in t_trans_cfg['pos_emb'] else False

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
                mha=mha,
                Aa=str(add_Aa)
            )
            self.t_trans_enc_layers = torch.nn.ModuleList(
                [copy.deepcopy(t_trans_enc_layer)
                 for _ in range(t_trans_cfg['num_layers'])])

            if 'res' in trans_seq:
                res_trans_enc_layers = torch.nn.ModuleDict(
                    {
                        'res_norm': nn.LayerNorm(
                            t_trans_dim, eps=t_trans_cfg['layer_norm_eps']),
                        'res_dropout': nn.Dropout(p=res_dropout),
                    }
                )
                self.res_trans_enc_layers = torch.nn.ModuleList(
                    [copy.deepcopy(res_trans_enc_layers)
                     for _ in range(t_trans_cfg['num_layers'])])

            pos_enc = str(pos_enc)
            if pos_enc == 'True' or pos_enc == 'original':
                self.t_pos_encoder = PositionalEncoding(t_trans_dim)
            elif pos_enc == 'cossin':
                self.t_pos_encoder = CosSinPositionalEncoding(t_trans_dim)
            else:
                self.t_pos_encoder = lambda x: x

        # 4. transformer (spatial)
        if s_trans_cfg is not None:
            s_trans_dim = s_trans_cfg['model_dim'] * trans_len
            s_trans_cfg['model_dim'] = s_trans_dim
            s_trans_cfg['ffn_dim'] *= trans_len

            add_A = str(add_A)
            if add_A == 'True' or add_A == 'Empty':
                m_dict = {}
                for a_i in range(self.num_subset):
                    if add_A == 'True':
                        m_dict[f'subset{a_i}'] = copy.deepcopy(
                            TransformerEncoderLayerExtV2(
                                cfg=s_trans_cfg,
                                mha=mha,
                                A=self.graph.A[a_i].transpose((1, 0)) if invert_A else self.graph.A[a_i],  # noqa
                                Aa=str(add_Aa)
                            )
                        )
                    elif add_A == 'Empty':
                        m_dict[f'subset{a_i}'] = copy.deepcopy(
                            TransformerEncoderLayerExtV2(
                                cfg=s_trans_cfg,
                                mha=mha,
                                A=None,
                                Aa=str(add_Aa)
                            )
                        )
                s_trans_enc_layer = torch.nn.ModuleDict(m_dict)
                s_trans_enc_layer.update(
                    {
                        'sa_norm': nn.LayerNorm(
                            s_trans_dim, eps=s_trans_cfg['layer_norm_eps']),
                        # 'sa_fc': nn.Linear(s_trans_dim, s_trans_dim)
                        'sa_dropout': nn.Dropout(p=sa_dropout),
                    }
                )
            elif add_A == 'False' or add_A == 'None':
                assert 'v0' in self.trans_seq, 'v0 not in self.trans_seq'
                s_trans_enc_layer = TransformerEncoderLayerExtV2(
                    cfg=s_trans_cfg,
                    mha=mha,
                    A=None,
                    Aa=str(add_Aa)
                )
            else:
                raise ValueError()

            self.s_trans_enc_layers = torch.nn.ModuleList(
                [copy.deepcopy(s_trans_enc_layer)
                    for _ in range(s_trans_cfg['num_layers'])]
            )

            pos_enc = str(pos_enc)
            if pos_enc == 'True' or pos_enc == 'original':
                self.s_pos_encoder = PositionalEncoding(s_trans_dim)
            elif pos_enc == 'cossin':
                self.s_pos_encoder = CosSinPositionalEncoding(s_trans_dim)
            else:
                self.s_pos_encoder = lambda x: x

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

        # 6. s cls token
        if add_s_cls and self.cls_token is not None:
            self.s_cls_token = nn.Parameter(torch.randn(1, 1, s_trans_dim))
            self.s_t_trans_enc_layer = torch.nn.ModuleDict(
                {
                    'st_linear1': nn.Linear(s_trans_dim*num_person,
                                            t_trans_dim),
                    'st_linear2': nn.Linear(t_trans_dim, t_trans_dim),
                    'st_dropout1': nn.Dropout(p=0.2),
                    'st_dropout2': nn.Dropout(p=0.2),
                    'st_act': nn.GELU(),
                    'st_norm': nn.LayerNorm(
                        t_trans_dim, eps=t_trans_cfg['layer_norm_eps']),
                }
            )
        else:
            self.s_cls_token = None

    def forward(self, x):
        if self.m_mask:
            if self.cls_token is not None:
                t = self.t_trans_enc_layers[0].self_attn.pos_emb.tokens - 1
            else:
                t = self.t_trans_enc_layers[0].self_attn.pos_emb.tokens
            self.m_b_mask = x.sum((1, 2, 3)) > 0   # n,m
            self.m_b_mask = self.m_b_mask.repeat(t//self.num_person, 1, 1)
            self.m_b_mask = self.m_b_mask.permute(1, 2, 0)  # n,m,t
            self.m_b_mask = self.m_b_mask.reshape(x.size(0), -1, 1)  # n,mt, 1
            if self.cls_token is not None:
                device = self.m_b_mask.get_device()
                if device == -1:
                    device = 'cpu'
                cls_token = torch.ones(
                    (x.size(0), 1, 1), dtype=bool,
                    device='cpu' if device == -1 else device)
                self.m_b_mask = torch.cat([cls_token, self.m_b_mask], dim=1)
        return super().forward(x)

    def forward_temporal_transformer(self,
                                     x: torch.Tensor,
                                     size: list,
                                     layer: torch.nn.Module,
                                     attn: list,
                                     pe: bool = False):
        N, C, T, V, M = size
        x = x.reshape(N, -1, V*C)  # n,1+mt,vc
        if self.m_mask:
            x = x * self.m_b_mask
        out = layer(x)  # x, attn, pe
        if self.need_attn:
            attn.append((out[1], out[2]) if pe else out[1])
        return out[0], attn

    def forward_spatial_Aa_trans(self,
                                 x: torch.Tensor,
                                 size: list,
                                 layers: Union[torch.nn.ModuleDict,
                                               torch.nn.ModuleList],
                                 attn: list,
                                 pe: bool = False,
                                 mode: str = 'v1'):
        assert mode in ['v0', 'v1', 'v2'], f"{mode} is not supported."
        N, C, T, V, M = size
        if self.cls_token is not None:
            x0 = x[:, 0:1, :]  # n,1,vc
            x1 = x[:, 1:, :]  # n,mt,vc
        else:
            x1 = x[:, :, :]  # n,mt,vc
        x1 = x1.view(N, M, T, V, C).permute(0, 1, 3, 2, 4).contiguous()
        x1 = x1.reshape(N*M, V, T*C)  # nm,v,tc
        if self.s_cls_token is not None:
            s_cls_tokens = self.s_cls_token.repeat(x1.size(0), 1, 1)
            x1 = torch.cat((s_cls_tokens, x1), dim=1)
        x1 = self.s_pos_encoder(x1)
        x_l = []
        if mode == 'v0':
            for s_name, s_layer in layers:
                out = s_layer(x1)
                x1 = out[0]
            if self.need_attn:
                attn.append((out[1], out[2]) if pe else out[1])
        elif mode == 'v1' or mode == 'v2':
            for s_name, s_layer in layers.items():
                if 'subset' not in s_name:
                    continue
                out = s_layer(x1, global_attn=s_layer.PA, alpha=s_layer.alpha)
                x_l.append(out[0])
                if self.need_attn:
                    attn.append((out[1], out[2]) if pe else out[1])
            x_l = torch.stack(x_l, dim=0).sum(dim=0)
        if mode == 'v1':
            x1 = layers['sa_dropout'](x_l)  # dropout
            x1 = layers['sa_norm'](x1)  # norm
        elif mode == 'v2':
            x1 = x1 + layers['sa_dropout'](x_l)  # dropout
            x1 = layers['sa_norm'](x1)  # norm
        if self.s_cls_token is not None:
            xs = x1[:, 0:1, :]
            xs = xs.view(N, M, 1, T, C).reshape(N, 1, -1)  # n,1,mtc
            xs = self.s_t_trans_enc_layer['st_linear1'](xs)
            xs = self.s_t_trans_enc_layer['st_act'](xs)
            xs = self.s_t_trans_enc_layer['st_dropout1'](xs)
            xs = self.s_t_trans_enc_layer['st_linear2'](xs)
            xs = self.s_t_trans_enc_layer['st_dropout2'](xs)
            xs = self.s_t_trans_enc_layer['st_norm'](xs)
            x0 += xs
            x1 = x1[:, 1:, :]
        x1 = x1.view(N, M, V, T, C).permute(0, 1, 3, 2, 4).contiguous()
        x1 = x1.reshape(N, M*T, V*C)  # n,mt,vc
        if self.cls_token is not None:
            x1 = torch.cat([x0, x1], dim=1)  # n,1+mt,vc
        return x1, attn

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
                s_trans_enc_layers = self.s_trans_enc_layers[i]
            else:
                s_trans_enc_layers = [(None, self.s_trans_enc_layers[i])]

            if 'v0' in self.trans_seq:
                # single s trans
                mode = 'v0'
            elif 'v1' in self.trans_seq:
                # 3 s trans -> sum -> norm -> dropout
                mode = 'v1'
            elif 'v2' in self.trans_seq:
                # 3 s trans -> sum -> res -> norm -> dropout
                mode = 'v2'

            # Spatial
            x1, attn[0] = self.forward_spatial_Aa_trans(
                x=x,
                size=[N, C, T, V, M],
                layers=s_trans_enc_layers,
                attn=attn[0],
                pe=self.rel_emb_s,
                mode=mode
            )

            # Temporal
            x2, attn[1] = self.forward_temporal_transformer(
                x=x if 'parallel' in self.trans_seq else x1,
                size=[N, C, T, V, M],
                layer=self.t_trans_enc_layers[i],
                attn=attn[1],
                pe=self.rel_emb_t
            )

            if 'parallel' in self.trans_seq:
                if 'add' in self.trans_seq:
                    x2 += x1
                else:
                    raise ValueError()

            if 'res' in self.trans_seq:
                # Residual
                x = x + self.res_trans_enc_layers[i]['res_dropout'](x2)
                x = self.res_trans_enc_layers[i]['res_norm'](x)
            else:
                x = x2

        x = x.reshape(N, -1, V*C)  # n,mt,vc
        if 'CLS' in self.classifier_type:
            x = x[:, 0, :]  # n,vc
        elif self.classifier_type == 'CLS-POOL':
            x = x[:, 0, :]  # n,vc
        elif 'GAP' in self.classifier_type:
            x = x.mean(1)  # n,vc
        else:
            raise ValueError("Unknown classifier_type")

        if 'POOL' in self.classifier_type:
            x = self.cls_pool_fc(x)
            x = self.cls_pool_act(x)

        return x, attn


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph,
                  model_layers=101,
                  t_trans_cfg={
                      'num_heads': 25,
                      'model_dim': 16,
                      'ffn_dim': 64,
                      'dropout': 0.2,
                      'activation': 'gelu',
                      'prenorm': False,
                      'num_layers': 3,
                      'pos_emb': 'rel-shared',
                      'length': 201,
                  },
                  s_trans_cfg={
                      'num_heads': 1,
                      'model_dim': 16,
                      'ffn_dim': 64,
                      'dropout': 0.2,
                      'activation': 'gelu',
                      'prenorm': False,
                      'num_layers': 3,
                      'pos_emb': 'rel-shared',
                      'length': 26,
                  },
                  trans_seq='sa-t-res-v2',
                  add_A='Empty',
                  add_Aa='False',
                  add_s_cls=True,
                  kernel_size=3,
                  pad=False,
                  pos_enc=None,
                  classifier_type='CLS-POOL'
                  )
    # print(model)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print([i[0] for i in model.named_parameters() if 'PA' in i[0]])
    # print(model(torch.ones((1, 3, 300, 25, 2))))
    print(model(torch.rand((1, 3, 300, 25, 2))))
