import torch
from torch import nn
from torchinfo import summary
from torch.nn.functional import _in_projection_packed, _in_projection
from torch.nn.functional import linear, pad, softmax, dropout
import warnings

import copy
import numpy as np
import math
from typing import Optional, Tuple

from model.aagcn import import_class
from model.aagcn import conv_init
from model.aagcn import bn_init
from model.aagcn import batch_norm_2d
from model.aagcn import GCNUnit
from model.aagcn import AdaptiveGCN
from model.aagcn import BaseModel


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


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    # if attn_mask is not None:
    #     attn += attn_mask
    attn = softmax(attn, dim=-1)
    if attn_mask is not None:
        attn += attn_mask
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def multi_head_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: torch.Tensor,
    in_proj_bias: Optional[torch.Tensor],
    bias_k: Optional[torch.Tensor],
    bias_v: Optional[torch.Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: torch.Tensor,
    out_proj_bias: Optional[torch.Tensor],
    training: bool = True,
    key_padding_mask: Optional[torch.Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[torch.Tensor] = None,
    k_proj_weight: Optional[torch.Tensor] = None,
    v_proj_weight: Optional[torch.Tensor] = None,
    static_k: Optional[torch.Tensor] = None,
    static_v: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias,
                bias_k, bias_v, out_proj_weight, out_proj_bias)
    if torch.has_torch_function(tens_ops):
        return torch.handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"  # noqa
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * \
        num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"  # noqa
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used  # noqa
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"  # noqa
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"  # noqa

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed(
            query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"  # noqa
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"  # noqa
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"  # noqa
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(
            query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)  # noqa

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")  # noqa
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or \
                attn_mask.dtype == torch.bool, \
               f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"  # noqa
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")  # noqa
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")  # noqa
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")  # noqa
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed  # noqa
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"  # noqa
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"  # noqa
        k = static_k
    if static_v is None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed  # noqa
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"  # noqa
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"  # noqa
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat(
            [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)  # noqa
        v = torch.cat(
            [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)  # noqa
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"  # noqa
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(
        0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(
            bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttentionExt(nn.MultiheadAttention):
    """MHA extension"""

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False,
                 kdim=None,
                 vdim=None,
                 batch_first=False,
                 device=None,
                 dtype=None) -> None:
        super(MultiheadAttentionExt, self).__init__(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
            bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
            kdim=kdim, vdim=vdim, batch_first=batch_first, device=device,
            dtype=dtype
        )

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0)
                                 for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


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
                      'num_layers': 2
                  },
                  s_trans_cfg={
                      'num_heads': 2,
                      'model_dim': 16,
                      'ffn_dim': 64,
                      'dropout': 0,
                      'activation': 'gelu',
                      'prenorm': False,
                      'num_layers': 2
                  },
                  trans_seq='sa-t-res',
                  #   add_A=True,
                  #   add_Aa=True,
                  kernel_size=3,
                  pad=False,
                  pos_enc='cossin'
                  )
    # print(model)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print([i[0] for i in model.named_parameters() if 'PA' in i[0]])
    model(torch.ones((3, 3, 300, 25, 2)))
