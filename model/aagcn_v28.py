import torch
from torch import nn
from torchinfo import summary

import copy
import numpy as np
import math
from typing import Optional

from DeBERTa import deberta

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
                                    pad=pad,
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
# ------------------------------------------------------------------------------
def default_transformer_config():
    config = deberta.ModelConfig()
    config.attention_probs_dropout_prob = 0.2
    config.hidden_act = "gelu"
    config.hidden_dropout_prob = 0.2
    config.hidden_size = 16
    config.initializer_range = 0.02
    config.intermediate_size = 64
    config.max_position_embeddings = 201
    config.layer_norm_eps = 1e-7
    config.num_attention_heads = 2
    config.num_hidden_layers = 3
    config.type_vocab_size = 0
    config.vocab_size = -1  # not used

    config.relative_attention = True
    config.position_buckets = 25  # the relative attn map span
    config.norm_rel_ebd = "layer_norm"  # relative embedding norm
    config.share_att_key = True  # whether to share the proj mat for pos and context attention caöculation.  # noqa
    config.pos_att_type = "p2c | c2p"  # p2p possible also

    config.conv_kernel_size = 3  # whether to use conv in the first layer
    config.conv_act = "gelu"

    config.max_relative_positions = -1  # if -1, uses max_position_embeddings
    config.position_biased_input = False  # whether to add PE to input
    config.attention_head_size = 200

    return config


def update_transformer_config(cfg: dict = None):
    if cfg is None:
        return default_transformer_config()
    else:
        def_cfg = default_transformer_config()
        for k, v in cfg.items():
            setattr(def_cfg, k, v)
        return def_cfg


def transformer_config_checker(cfg: dict):
    trans_cfg_names = [
        'attention_probs_dropout_prob',
        'hidden_act',
        'hidden_dropout_prob',
        'hidden_size',
        'initializer_range',
        'intermediate_size',
        'max_position_embeddings',
        'layer_norm_eps',
        'num_attention_heads',
        'num_hidden_layers',
        'type_vocab_size',
        'padding_idx',
        'vocab_size',
        # Extras:
        'relative_attention',
        'position_buckets',
        'norm_rel_ebd',
        'share_att_key',
        'pos_att_type',
        'conv_kernel_size',
        'conv_act',
        'max_relative_positions',
        'position_biased_input',
        'attention_head_size'
    ]
    for x in cfg.__dict__.keys():
        assert f'{x}' in trans_cfg_names, f'{x} not in transformer config'


# class DeBERTa(deberta.DeBERTa):
class DeBERTa(torch.nn.Module):
    def __init__(self, config=None, pre_trained=None):
        super().__init__()
        state = None
        # if pre_trained is not None:
        #     state, model_config = load_model_state(pre_trained)
        #     if config is not None and model_config is not None:
        #         for k in config.__dict__:
        #         if k not in ['hidden_size',
        #                      'intermediate_size',
        #                      'num_attention_heads',
        #                      'num_hidden_layers',
        #                      'vocab_size',
        #                      'max_position_embeddings']:
        #             model_config.__dict__[k] = config.__dict__[k]
        #     config = copy.copy(model_config)
        # self.embeddings = BertEmbeddings(config)
        # attn -> linear residual -> linear -> linear residual
        self.encoder = deberta.BertEncoder(config)
        self.config = config
        self.pre_trained = pre_trained
        self.apply_state(state)

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                output_all_encoded_layers=True,
                position_ids=None,
                return_att=False):
        """
        input_ids is a tensor with size n,m,c (batch, length, channel)
        attention_mask masks the attention map.
        token_type_ids used to label segments.
        """

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids[:, :, 0])
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids[:, :, 0])

        # embedding_output = self.embeddings(
        #     input_ids.to(torch.long),
        #     token_type_ids.to(torch.long),
        #     position_ids,
        #     attention_mask
        # )
        encoded_layers = self.encoder(
            input_ids,
            attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            return_att=return_att
        )
        if return_att:
            encoded_layers, att_matrixs = encoded_layers

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1:]

        if return_att:
            return encoded_layers, att_matrixs
        return encoded_layers

    def apply_state(self, state=None):
        if self.pre_trained is None and state is None:
            return
        if state is None:
            state, config = deberta.load_model_state(self.pre_trained)
            self.config = config

        def key_match(key, s):
            c = [k for k in s if key in k]
            assert len(c) == 1, c
            return c[0]
        current = self.state_dict()
        for c in current.keys():
            current[c] = state[key_match(c, state.keys())]
        self.load_state_dict(current)


# ------------------------------------------------------------------------------
# Network
# - uses deberta
# - DEBERTA: DECODING-ENHANCED BERT WITH DISENTANGLED ATTENTION
# - attention PE, context-context, position-context, context-position.
# - from v27 and v17
# - only temporal attention
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
                 trans_cfg: Optional[dict] = None,
                 add_A: bool = False,
                 pos_enc: str = 'True',
                 classifier_type: str = 'CLS',
                 model_layers: int = 10
                 ):

        super().__init__(num_class, num_point, num_person,
                         in_channels, drop_out, adaptive, gbn_split)

        trans_cfg = update_transformer_config(trans_cfg)
        transformer_config_checker(trans_cfg)

        self.need_attn = need_attn

        # 1. joint graph
        self.init_graph(graph, graph_args)

        # 2. aagcn layer
        def _TCNGCNUnit(_in, _out, stride=kernel_size, padding=pad,
                        residual=True):
            return TCNGCNUnit(_in,
                              _out,
                              self.graph.A,
                              num_subset=num_subset,
                              kernel_size=kernel_size,
                              stride=stride,
                              pad=padding,
                              residual=residual,
                              adaptive=self.adaptive_fn,
                              attention=attention,
                              gbn_split=gbn_split)

        self.init_model_backbone(model_layers=model_layers,
                                 tcngcn_unit=_TCNGCNUnit,
                                 output_channel=trans_cfg.hidden_size)

        # 4. transformer (spatial)
        trans_cfg.hidden_size = trans_cfg.hidden_size*num_point
        self.deberta = DeBERTa(config=trans_cfg)

        # 5. classifier
        self.classifier_type = classifier_type
        if classifier_type == 'CLS':
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, trans_cfg.hidden_size))
        else:
            self.cls_token = None

        self.init_fc(trans_cfg.hidden_size, num_class)

    def forward_postprocess(self, x: torch.Tensor, size: torch.Size):
        N, _, _, V, M = size
        _, C, T, _ = x.size()
        x = x.view(N, M, C, T, V).permute(0, 1, 3, 4, 2).contiguous()  # n,m,t,v,c  # noqa
        x = x.view(N, M*T, C*V)  # n,mt,vc

        if self.cls_token is not None:
            cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
            x = torch.cat((cls_tokens, x), dim=1)

        encoding = self.deberta(x, return_att=self.need_attn)

        if self.need_attn:
            x, attn = encoding
        else:
            x, attn = encoding, []

        if self.classifier_type == 'CLS':
            x = x[-1][:, 0, :]  # n,vc
        elif self.classifier_type == 'GAP':
            x = x[-1].mean(1)  # n,vc
        else:
            raise ValueError("Unknown classifier_type")

        return x, attn


if __name__ == '__main__':
    graph = 'graph.ntu_rgb_d.Graph'
    model = Model(graph=graph,
                  model_layers=101,
                  #   s_trans_cfg={
                  #       'num_heads': 2,
                  #       'model_dim': 16,
                  #       'ffn_dim': 64,
                  #       'dropout': 0,
                  #       'activation': 'gelu',
                  #       'prenorm': False,
                  #       'num_layers': 3
                  #   },
                  kernel_size=3,
                  pad=False,
                  pos_enc='cossin'
                  )
    print(model)
    # summary(model, (1, 3, 300, 25, 2), device='cpu')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print([i[0] for i in model.named_parameters() if 'PA' in i[0]])
    model(torch.ones((3, 3, 300, 25, 2)))
