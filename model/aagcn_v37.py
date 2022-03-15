import torch
from torch import nn
from torch.nn.functional import gelu

import copy
import numpy as np
import math
from typing import Optional, OrderedDict, Union

from model.aagcn import TCNGCNUnit
from model.aagcn import BaseModel

from model.module.multiheadattention import MultiheadAttention
from model.module.crossattention import Transformer
from model.module.crossattention import CrossTransformer
from model.module.crossattention import CrossTransformerIdentity


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


def ca_transformer_config_checker(cfg: dict):
    trans_cfg_names = [
        'depth',
        'sm_dim',
        'sm_heads',
        'sm_dim_head',
        'sm_dropout',
        'lg_dim',
        'lg_heads',
        'lg_dim_head',
        'lg_dropout',
        'num_layers',
    ]
    for x in cfg:
        assert f'{x}' in trans_cfg_names, f'{x} not in transformer ca config'


def transformer_config_checker(cfg: dict):
    trans_cfg_names = [
        'dim',
        'depth',
        'heads',
        'dim_head',
        'mlp_dim',
        'dropout',
        'num_layers',
        'length',
        'pos_emb'
    ]
    for x in cfg:
        assert f'{x}' in trans_cfg_names, f'{x} not in transformer config'


# ------------------------------------------------------------------------------
"""
Network
- uses s and t transformers with cross attention transformer

n, m, t, v, c(raw)

1. nm, t, vc(corr-time-per-m, joint stacked)
2. n, mt, vc(corr-time-inter-m, joint stacked)(vvvvvv)
3. nv, t, mc(corr-t-per-joimt, m stacked)
4. n, vt, mc(corr-joint-inter-time, m stacked)
- (seems to be overkill, corr across time is there in 2.)
5. n, mv, tc(corr-joint-inter-m, time stacked)(vvvvv)
6. n, t, mvc(corr-time-overall, global-ish)(vvvvv)
7. n, v, mtc(corr-joint-overall, global-ish)(vvvvv)
- 1. gives correlation over time per m, (using vc per m).
- 2. gives correlation over time per m and also inter m, (using vc).
- 3. gives correlation over time per v, (using mc per v).
- 4. gives correlation over time per v and also inter v, (using mc).
- 5. gives correlation over V per m and also inter m, (using mc).
- 6. gives correlation over time, using mvc.
- 1. assumes no corr in time between m, but 2. does.
- 2. is useful if most actions have correlations between m.
- 3. assumes no corr in time between v.
- 4. is useful if most actions have correlations between v.
- 6. is person and joint agnostic.

nm, v, tc(corr-joint-per-m, time stacked)
n, mv, tc(corr-joint-inter-m, time stacked)
n, v, mtc(corr-joint-overall, global-ish)
- same logic as before but with v instead of t.

n, m, t, v, c(raw)
n, t, mvc(corr-time-overall, global-ish)
n, m, tvc(m=2, learn corr?)
n, v, mtc(corr-joint-overall, global-ish)
n, mt, vc(corr-time-inter-m, joint stacked)
n, mv, tc(corr-joint-inter-m, time stacked)
n, tv, mc(corr-joint/time-inter-time/joint, m stacked)
n, mtv, c(corr-m-time-joint, too long?)
nm, t, vc(corr-time-per-m, joint stacked)
nm, v, tc(corr-joint-per-m, time stacked)
nt, m, vc(corr-m-per-t, learn corr?)
nt, v, mc(corr-joint-per-t, m stacked)
nv, m, tc(corr-m-per-joint, learn corr?)
nv, t, mc(corr-t-per-joimt, m stacked)
nmv, t, c(corr-time-channel, joint+person independent)

n, t, mvc(corr-time-overall, global-ish)
n, v, mtc(corr-joint-overall, global-ish)
nm, t, vc(corr-time-per-m, joint stacked)
nm, v, tc(corr-joint-per-m, time stacked)
n, mt, vc(corr-time-inter-m, joint stacked)
n, mv, tc(corr-joint-inter-m, time stacked)

nmv, t, c(corr-time-channel, joint+person independent)

nv, t, mc(corr-t-per-joimt, m stacked)
nt, v, mc(corr-joint-per-t, m stacked)
"""
# ------------------------------------------------------------------------------


class Model(BaseModel):
    def __init__(self,
                 # aagcn args
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
                 # additional aagcn args
                 kernel_size: int = 9,
                 pad: bool = True,
                 backbone_dim: int = 16,
                 model_layers: int = 10,
                 # transformer args
                 t_trans_cfg: Optional[dict] = None,
                 s_trans_cfg: Optional[dict] = None,
                 c_trans_cfg: Optional[dict] = None,
                 trans_mode: str = 'n-t-mvc',
                 pos_enc: str = 'True',
                 # global A args
                 add_A: Optional[str] = None,
                 add_alpha_A: Optional[str] = None,
                 invert_A: bool = False,
                 # transformer classifier
                 add_s_cls_token: bool = True,
                 add_t_cls_token: bool = True,
                 classifier_type: str = 'CLS',
                 ):

        super().__init__(num_class, num_point, num_person,
                         in_channels, drop_out, adaptive, gbn_split)

        # 0. setup
        if t_trans_cfg is not None:
            transformer_config_checker(t_trans_cfg)
        if s_trans_cfg is not None:
            transformer_config_checker(s_trans_cfg)
        if c_trans_cfg is not None:
            ca_transformer_config_checker(c_trans_cfg)

        assert t_trans_cfg['num_layers'] == s_trans_cfg['num_layers']
        assert t_trans_cfg['num_layers'] % c_trans_cfg['num_layers'] == 0

        self.num_subset = num_subset
        self.trans_mode = trans_mode

        # self.rel_emb_s = True if 'rel' in s_trans_cfg['pos_emb'] else False
        # self.rel_emb_t = True if 'rel' in t_trans_cfg['pos_emb'] else False

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
            self.t_trans_enc = nn.ModuleDict(
                {
                    f'block{i+1}': Transformer(**t_trans_cfg)
                    for i in range(t_trans_cfg['num_layers'])
                }
            )

            pos_enc = str(pos_enc)
            if pos_enc == 'True' or pos_enc == 'original':
                self.t_pos_encoder = PositionalEncoding(t_trans_cfg['dim'])
            elif pos_enc == 'cossin':
                self.t_pos_encoder = CosSinPositionalEncoding(
                    t_trans_cfg['dim'])
            else:
                self.t_pos_encoder = lambda x: x

        # 4. transformer (spatial)
        if s_trans_cfg is not None:
            self.s_trans_enc = nn.ModuleDict(
                {
                    f'block{i+1}': Transformer(**s_trans_cfg)
                    for i in range(s_trans_cfg['num_layers'])
                }
            )

            pos_enc = str(pos_enc)
            if pos_enc == 'True' or pos_enc == 'original':
                self.s_pos_encoder = PositionalEncoding(s_trans_cfg['dim'])
            elif pos_enc == 'cossin':
                self.s_pos_encoder = CosSinPositionalEncoding(
                    s_trans_cfg['dim'])
            else:
                self.s_pos_encoder = lambda x: x

        # 5. classifier
        if add_s_cls_token:
            self.s_cls_token = nn.Parameter(
                torch.randn(1, 1, s_trans_cfg['dim']))
        else:
            self.s_cls_token = None
        if add_t_cls_token:
            self.t_cls_token = nn.Parameter(
                torch.randn(1, 1, t_trans_cfg['dim']))
        else:
            self.t_cls_token = None

        output_dim = t_trans_cfg['dim'] + s_trans_cfg['dim']
        if 'POOL' in classifier_type:
            self.cls_pool_fc = nn.Linear(output_dim, output_dim)
            self.cls_pool_act = nn.Tanh()
        else:
            self.cls_pool_fc = lambda x: x
            self.cls_pool_act = lambda x: x
        self.init_fc(output_dim, num_class)

        # 7. cross attention
        # sm = longer token length
        ratio = s_trans_cfg['num_layers']//c_trans_cfg['num_layers']
        assert ratio >= 1
        self.cross_attn_enc = nn.ModuleDict({})
        for i in range(s_trans_cfg['num_layers']):
            if (i+1) % ratio == 0:
                self.cross_attn_enc.update(
                    {
                        f'block{i+1}': CrossTransformer(**c_trans_cfg)
                    }
                )
            else:
                self.cross_attn_enc.update(
                    {
                        f'block{i+1}': CrossTransformerIdentity()
                    }
                )

    def forward_postprocess(self, x: torch.Tensor, size: torch.Size):
        N, _, _, V, M = size
        _, C, T, _ = x.size()

        x1 = x.view(N, M, C, T, V).permute(0, 4, 1, 3, 2).contiguous()  # n,v,m,t,c  # noqa
        x1 = x1.reshape(N, V, M*T*C)  # n,v,mtc
        if self.s_cls_token is not None:
            s_cls_tokens = self.s_cls_token.repeat(x1.size(0), 1, 1)
            x1 = torch.cat((s_cls_tokens, x1), dim=1)  # n,v+1,mtc
        x1 = self.s_pos_encoder(x1)

        x2 = x.view(N, M, C, T, V).permute(0, 3, 1, 4, 2).contiguous()  # n,t,m,v,c  # noqa
        x2 = x2.reshape(N, T, M*V*C)  # n,t,mvc
        if self.t_cls_token is not None:
            t_cls_token = self.t_cls_token.repeat(x2.size(0), 1, 1)
            x2 = torch.cat((t_cls_token, x2), dim=1)  # n,t+1,mvc
        x2 = self.t_pos_encoder(x2)

        attn_list = [[], [], []]

        for (_, t_block), (_, s_block), (_, c_block) in zip(
            self.t_trans_enc.items(),
            self.s_trans_enc.items(),
            self.cross_attn_enc.items()
        ):
            # Spatial
            x1, attn = s_block(x1)  # n,v+1,mtc
            attn_list[0].append(attn)
            # Temporal
            x2, attn = t_block(x2)  # n,t+1,mvc
            attn_list[1].append(attn)
            # cross attention
            x2, x1, attn = c_block(x2, x1)
            attn_list[2].append(attn)

        x1 = x1[:, 0, :]  # n,mtc
        x2 = x2[:, 0, :]  # n,mvc
        x = torch.cat([x1, x2], dim=-1)  # n,mtc+mvc

        x = self.cls_pool_fc(x)
        x = self.cls_pool_act(x)

        return x, attn_list


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph,
                  model_layers=101,
                  t_trans_cfg={
                      'dim': 16*2*25,
                      'depth': 1,
                      'heads': 25,
                      'dim_head': 16,
                      'mlp_dim': 64*2*25,
                      'dropout': 0.2,
                      'pos_emb': 'rel-shared',
                      'length': 101,
                      'num_layers': 3,
                  },
                  s_trans_cfg={
                      'dim': 16*100*2,
                      'depth': 3,
                      'heads': 1,
                      'dim_head': 26*16,
                      'mlp_dim': 64*100*2,
                      'dropout': 0.2,
                      'pos_emb': 'rel-shared',
                      'length': 26,
                      'num_layers': 3,
                  },
                  c_trans_cfg={
                      'depth': 1,
                      'sm_dim': 16*2*25,
                      'sm_heads': 16,
                      'sm_dim_head': 2*25,
                      'sm_dropout': 0.2,
                      'lg_dim': 16*2*100,
                      'lg_heads': 1,
                      'lg_dim_head': 16*2*100,
                      'lg_dropout': 0.2,
                      'num_layers': 1,
                  },
                  add_s_cls_token=True,
                  add_t_cls_token=True,
                  kernel_size=3,
                  pad=False,
                  pos_enc=None,
                  classifier_type='CLS-POOL'
                  )

    print(model)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print([i[0] for i in model.named_parameters() if 'PA' in i[0]])
    # print(model(torch.ones((1, 3, 300, 25, 2))))
    print(model(torch.rand((1, 3, 300, 25, 2)))[0])
