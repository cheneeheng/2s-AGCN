import torch
from torch import nn
from torch import Tensor
from torch.nn import Module as PyTorchModule

try:
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
except ImportError:
    print("Warning: fvcore is not found")

import math
from typing import Tuple, Optional, Union, Type, List

from model.resource.common_ntu import *
from model.layers import Module
from model.layers import Conv
from model.layers import residual as res
from model.layers import fuse_features
from model.layers import null_fn
from model.layers import pad_zeros
from model.layers import get_activation_fn
from model.layers import get_normalization_fn
from model.layers import Transformer

from utils.utils import *
from utils.parser import get_parser
from utils.parser import load_parser_args_from_config

from model.architecture.sgn.blocks import Embedding
from model.architecture.sgn.blocks import SemanticEmbedding
from model.architecture.sgn.blocks import FeatureExtractor

T1 = Type[PyTorchModule]
T2 = List[Optional[Type[PyTorchModule]]]
T3 = Union[List[int], int]

EMB_MODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 101, 1001]

# Maxpooling ---
# 0 no pool
# 1 pool
# 2 pool with indices
POOLING_MODES = [0, 1, 2]


class SGN(PyTorchModule):

    # CONSTANTS
    emb_modes = EMB_MODES  # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    c1, c2, c3, c4 = c1, c2, c3, c4
    # g_activation_fn = nn.Identity
    g_activation_fn = nn.Softmax

    def __init__(self,
                 num_class: int = 60,
                 num_point: int = 25,
                 num_segment: int = 20,
                 in_channels: int = 3,
                 bias: int = 1,
                 dropout: float = 0.0,  # classifier
                 dropout2d: float = 0.0,  # the rest
                 c_multiplier: Union[int, float, list] = 1,
                 norm_type: str = 'bn-pre',
                 act_type: str = 'relu',

                 input_position: int = 1,
                 input_velocity: int = 1,
                 semantic_joint: int = 1,
                 semantic_frame: int = 1,
                 semantic_class: int = 0,

                 input_emb_fusion: int = 1,
                 semantic_joint_fusion: int = 0,
                 semantic_frame_fusion: int = 1,
                 semantic_frame_location: int = 0,

                 spatial_maxpool: int = 1,
                 temporal_maxpool: int = 1,

                 spatial_mha_kwargs: Optional[dict] = None,
                 temporal_mha_kwargs: Optional[dict] = None,
                 ):
        super(SGN, self).__init__()

        # Base SGN args --------------------------------------------------------
        self.num_class = num_class
        self.num_point = num_point
        self.num_segment = num_segment
        self.in_channels = in_channels
        self.bias = bias
        self.dropout_fn = lambda: nn.Dropout2d(dropout2d)
        self.fc_dropout = nn.Dropout(dropout)

        if isinstance(c_multiplier, (int, float)):
            c_multiplier = [c_multiplier for _ in range(4)]
        elif isinstance(c_multiplier, list):
            c_multiplier = c_multiplier
        else:
            raise ValueError("Unknown c_multiplier type")
        self.c1 = to_int(self.c1 * c_multiplier[0])  # pos,vel,joint embed
        self.c2 = to_int(self.c2 * c_multiplier[1])  # G,gcn
        self.c3 = to_int(self.c3 * c_multiplier[2])  # gcn
        self.c4 = to_int(self.c4 * c_multiplier[3])  # gcn

        (self.normalization_fn_1d,
         self.normalization_fn) = get_normalization_fn(norm_type)
        self.prenorm = True if 'pre' in norm_type else False

        self.activation_fn = get_activation_fn(act_type)

        # Input + Semantics ----------------------------------------------------
        self.input_position = input_position
        self.input_velocity = input_velocity
        self.semantic_joint = semantic_joint
        self.semantic_frame = semantic_frame
        self.semantic_class = semantic_class
        assert self.input_position in self.emb_modes
        assert self.input_velocity in self.emb_modes
        assert self.semantic_joint in self.emb_modes
        assert self.semantic_frame in self.emb_modes
        if self.input_position == 0 and self.semantic_joint > 0:
            raise ValueError("input_position is 0 but semantic_joint is not")

        # 0: concat, 1: sum
        self.input_emb_fusion = input_emb_fusion
        # 0: concat, 1: sum
        self.semantic_joint_fusion = semantic_joint_fusion
        # 0: concat, 1: sum
        self.semantic_frame_fusion = semantic_frame_fusion  # UNUSED
        # 0 = add after GCN, 1 = add before GCN
        self.semantic_frame_location = semantic_frame_location
        assert self.semantic_frame_location in [0, 1]

        # Dynamic Representation -----------------------------------------------
        self.feature_extractor = FeatureExtractor(
            in_pos=self.input_position,
            in_vel=self.input_velocity,
            in_pos_emb_kwargs=dict(
                in_channels=self.in_channels,
                out_channels=self.c1,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                in_norm=self.normalization_fn_1d,
                num_point=self.num_point,
                mode=self.input_position,
            ),
            in_vel_emb_kwargs=dict(
                in_channels=self.in_channels,
                out_channels=self.c1,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                in_norm=self.normalization_fn_1d,
                num_point=self.num_point,
                mode=self.input_velocity,
            ),
            fusion=self.input_emb_fusion
        )

        # Semantic Embeddings --------------------------------------------------
        if self.input_emb_fusion == 0:
            sem_out = self.c2
        elif self.input_emb_fusion == 1:
            sem_out = self.c1

        self.semantic_embedding = SemanticEmbedding(
            num_point=self.num_point,
            num_segment=self.num_segment,
            sem_spa=self.semantic_joint,
            sem_tem=self.semantic_frame,
            sem_cls=self.semantic_class,
            sem_spa_emb_kwargs=dict(
                in_channels=self.num_point,
                out_channels=sem_out,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                num_point=self.num_point,
                mode=self.semantic_joint
            ),
            sem_tem_emb_kwargs=dict(
                in_channels=self.num_segment,
                out_channels=self.c3,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                num_point=self.num_point,
                mode=self.semantic_frame
            ),
            sem_cls_emb_kwargs=dict(
                in_channels=1,
                out_channels=self.c4,
                bias=self.bias,
                dropout=self.dropout_fn,
                activation=self.activation_fn,
                normalization=self.normalization_fn,
                num_point=self.num_point,
                mode=self.semantic_class
            )
        )

        # Transformers ---------------------------------------------------------
        self.spatial_mha = SpatialMHA(spatial_mha_kwargs)
        self.temporal_mha = TemporalMHA(temporal_mha_kwargs)

        # Maxpool --------------------------------------------------------------
        # 0 no pool, 1 pool, 2 pool with indices
        self.spatial_maxpool = spatial_maxpool
        self.temporal_maxpool = temporal_maxpool
        assert self.spatial_maxpool in POOLING_MODES
        assert self.temporal_maxpool in POOLING_MODES

        if self.spatial_maxpool == 0:
            self.smp = nn.Identity()
        elif self.spatial_maxpool == 1:
            self.smp = nn.AdaptiveMaxPool2d((1, self.num_segment))
        else:
            raise ValueError("Unknown spatial_maxpool")

        if self.temporal_maxpool == 0:
            self.tmp = nn.Identity()
        elif self.temporal_maxpool == 1:
            self.tmp = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError("Unknown temporal_maxpool")

        # Classifier ---------------------------------------------------------
        fc_in_ch = self.c4  # default
        if self.spatial_maxpool == 0 and self.temporal_maxpool == 0:
            fc_in_ch = fc_in_ch * self.num_segment * self.num_point
        if self.temporal_maxpool == 0:
            fc_in_ch = fc_in_ch * self.num_segment

        self.fc = nn.Linear(fc_in_ch, num_class)

        # Init weight ----------------------------------------------------------
        self.init_weight()

    def init_weight(self):
        """Follows the weight initialization from the original SGN codebase."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x: Tensor) -> Tuple[Tensor, list]:
        """Model forward pass.

        Args:
            x (Tensor): 3D joint in_pos of a sequence of skeletons.

        Returns:
            y (Tensor): logits.
            g (Tensor): Attention matrix for GCN.
        """
        bs, step, dim = x.shape
        x = x.view((bs, step, dim//self.in_channels, self.in_channels))  # n,t,v,c  # noqa
        x = x.permute(0, 3, 2, 1).contiguous()  # n,c,v,t

        # if x.shape[1] < self.in_channels:
        #     raise ValueError("tensor x has lower ch dim than self.in_channels")  # noqa
        # elif x.shape[-1] > self.in_channels:
        #     # print("tensor x has more ch dim than self.in_channels")
        #     # bs, step, dim = x.shape
        #     x = x[:, :self.in_channels, :, :]

        # Dynamic Representation -----------------------------------------------
        x, pos_emb, vel_emb = self.feature_extractor(x)
        assert x is not None

        # Joint and frame embeddings -------------------------------------------
        spa_emb, tem_emb, cls_emb = self.semantic_embedding(x)

        # Joint-level Module ---------------------------------------------------
        # xyz embeddings
        if spa_emb is None:
            x = x  # n,c,v,t
        else:
            # xyz emb (c) spa emb
            # xyz emb + spa emb
            x = fuse_features(x, spa_emb, self.semantic_joint_fusion)

        # temporal fusion pre gcn
        if self.semantic_frame > 0 and self.semantic_frame_location == 1:
            x = x + tem_emb

        # GCN ------------------------------------------------------------------
        spatial_featuremap, spatial_attn_list = self.spatial_mha(x)
        x = spatial_featuremap

        # Frame-level Module ---------------------------------------------------
        # temporal fusion post gcn
        if self.semantic_frame > 0 and self.semantic_frame_location == 0:
            x = x + tem_emb

        # spatial pooling
        x = self.smp(x)

        # temporal MLP
        temporal_featuremap, temporal_attn_list = self.temporal_mha(x)
        x = temporal_featuremap

        # temporal pooling
        y = self.tmp(x)

        if cls_emb is not None:
            y += cls_emb

        # Classification -------------------------------------------------------
        y = torch.flatten(y, 1)
        y = self.fc_dropout(y)
        y = self.fc(y)

        return (
            y,
            {
                'tem_emb': tem_emb,
                'spa_emb': spa_emb,
                'spatial_attn_list': spatial_attn_list,
                'temporal_attn_list': temporal_attn_list,
                'spatial_featuremap': spatial_featuremap,
                'temporal_featuremap': temporal_featuremap,
            }
        )


class DataNorm(PyTorchModule):
    def __init__(self, dim: int, normalization: T1 = nn.BatchNorm1d):
        super(DataNorm, self).__init__()
        self.bn = normalization(dim)  # channel dim * num_point

    def forward(self, x: Tensor) -> Tensor:
        bs, _, num_point, step = x.shape  # n,c,v,t
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_point, step).contiguous()
        return x


class Embedding(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: int = 0,
                 dropout: T1 = nn.Dropout2d,
                 activation: T1 = nn.ReLU,
                 normalization: T1 = nn.BatchNorm2d,
                 in_norm: Optional[T1] = None,
                 #  inter_channels: list = [0],
                 #  residual: int = 0,
                 num_point: int = 25,
                 mode: int = 1
                 ):
        super(Embedding, self).__init__(in_channels,
                                        out_channels,
                                        bias=bias,
                                        dropout=dropout,
                                        activation=activation,
                                        normalization=normalization)
        assert mode in EMB_MODES
        self.mode = mode

        if in_norm is not None:
            self.norm = DataNorm(self.in_channels * num_point, in_norm)
        else:
            self.norm = nn.Identity()

        if self.mode // 100 == 0:
            if self.mode == 1:
                ch_list = [self.in_channels,
                           self.out_channels, self.out_channels]
                k_list = [1, 1]
                residual = 0
            elif self.mode == 2:
                ch_list = [self.in_channels,
                           self.out_channels, self.out_channels]
                k_list = [1, 1]
                residual = 1
            elif self.mode == 3:
                ch_list = [self.in_channels,
                           self.out_channels, self.out_channels,
                           self.out_channels]
                k_list = [1, 1, 1]
                residual = 0
            elif self.mode == 4:
                ch_list = [self.in_channels,
                           self.out_channels, self.out_channels,
                           self.out_channels, self.out_channels]
                k_list = [1, 1, 1, 1]
                residual = 0
            elif self.mode == 11:
                ch_list = [self.in_channels,
                           self.out_channels, self.out_channels]
                k_list = [1, 3]
                residual = 0
            elif self.mode == 12:
                ch_list = [self.in_channels,
                           self.out_channels, self.out_channels]
                k_list = [3, 3]
                residual = 0

            self.num_layers = len(ch_list) - 1
            for i in range(self.num_layers):
                setattr(self, f'cnn{i+1}', Conv(ch_list[i],
                                                ch_list[i+1],
                                                kernel_size=k_list[i],
                                                padding=k_list[i]//2,
                                                bias=self.bias,
                                                activation=self.activation))
            for i in range(self.num_layers):
                setattr(self, f'res{i+1}', res(residual,
                                               ch_list[i],
                                               ch_list[i+1],
                                               self.bias))

        elif self.mode // 100 == 1:
            # bert style
            self.num_layers = 1
            self.cnn1 = Conv(self.in_channels,
                             self.out_channels,
                             bias=self.bias,
                             normalization=lambda: self.normalization(
                                 self.out_channels),
                             dropout=self.dropout)
            self.res1 = null_fn

    def forward(self, x: Tensor) -> Tensor:
        # x: n,c,v,t
        x = self.norm(x)
        for i in range(self.num_layers):
            x = getattr(self, f'cnn{i+1}')(x) + getattr(self, f'res{i+1}')(x)
        return x


class MHA(PyTorchModule):
    def __init__(self, **kwargs):
        super(MHA, self).__init__()
        if 'norm' in kwargs:
            self.transformer = Transformer(
                dim=kwargs.pop('d_model'),
                depth=kwargs.pop('num_layers'),
                heads=kwargs.pop('nhead'),
                dim_head=kwargs.pop('d_head'),
                dropout=kwargs.pop('dropout'),
                mlp_dim=kwargs.pop('dim_feedforward'),
                mlp_out_dim=kwargs.pop('dim_feedforward_output'),
                activation=kwargs.pop('activation'),
                norm=kwargs.pop('norm'),
                global_norm=kwargs.pop('global_norm'),
                **kwargs
            )
        else:
            self.num_layers = kwargs['num_layers']
            for i in range(self.num_layers):
                setattr(self,
                        f'layer{i+1}',
                        nn.TransformerEncoderLayer(
                            d_model=kwargs['d_model'],
                            nhead=kwargs['nhead'],
                            dim_feedforward=kwargs['dim_feedforward'],
                            dropout=kwargs['dropout'],
                            activation=kwargs['activation'],
                            layer_norm_eps=1e-5,
                            batch_first=True,
                        ))

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[list]]:
        if hasattr(self, 'transformer'):
            x, attn_list = self.transformer(x)
        else:
            attn_list = None
            for i in range(self.num_layers):
                x = getattr(self, f'layer{i+1}')(x)
        return x, attn_list


class SpatialMHA(MHA):
    def __init__(self, kwargs):
        super(SpatialMHA, self).__init__(**kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[list]]:
        # x: n,c,v,t
        N, _, V, T = x.shape
        x = x.permute(0, 3, 2, 1).contiguous()  # n,t,v,c
        x = x.reshape(N*T, V, -1)
        x, attn_list = super().forward(x)
        x = x.reshape(N, T, V, -1)
        x = x.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
        return x, attn_list


class TemporalMHA(MHA):
    def __init__(self, kwargs):
        super(TemporalMHA, self).__init__(**kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[list]]:
        # x: n,c,v,t
        N, _, V, T = x.shape
        x = x.permute(0, 3, 2, 1).contiguous()  # n,t,v,c
        x = x.reshape(N, T, -1)
        x, attn_list = super().forward(x)
        x = x.reshape(N, T, V, -1)
        x = x.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
        return x, attn_list


if __name__ == '__main__':

    batch_size = 1
    inputs = torch.ones(batch_size, 20, 75)
    # subjects = torch.ones(batch_size, 40, 1)

    # base_path = "/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15"
    # # file_path = "/221031120001_1head_256hdim_256out_nov_res_8head_16hdim_bn_rerun"  # noqa
    # # file_path = "/221124163001_8head_16hdim_128ffn_256out_8head_32hdim_256ffn_512out_qkv"  # noqa
    # file_path = "/221130163001_8head_16hdim_128ffn_256out_8head_32hdim_256ffn_512out_qkv_inputconcat"  # noqa
    # parser = get_parser()
    # parser.set_defaults(**{'config': base_path + file_path + "/config.yaml"})
    # args = load_parser_args_from_config(parser)
    # print(args.model_args)
    # model = SGN(**args.model_args)
    # model(inputs)
    # print(model)

    model = SGN(
        num_class=60,
        num_point=25,
        num_segment=20,
        in_channels=3,
        bias=1,
        dropout=0.0,  # classifier
        dropout2d=0.2,  # the rest
        c_multiplier=[1.0, 1.0, 1.0, 1.0],
        norm_type='bn',
        act_type='relu',
        input_position=1,
        input_velocity=1,
        semantic_joint=1,
        semantic_frame=1,
        input_emb_fusion=0,
        semantic_joint_fusion=1,
        semantic_frame_fusion=1,
        semantic_frame_location=0,
        spatial_mha_kwargs={
            'd_model': [128],
            'nhead': [1],
            'd_head': [256],
            'd_out': [256],
            # 'v_proj': False,
            # 'res_proj': True,
            'dim_feedforward': [256],
            'dim_feedforward_output': [256],
            'dropout': 0.1,
            'activation': "relu",
            'num_layers': 1,
            'norm': 'bn',
            'global_norm': False
        },
        temporal_mha_kwargs={
            'd_model': [256],
            'nhead': [8],
            'd_head': [16],
            'dim_feedforward': [256],
            'dim_feedforward_output': [512],
            'dropout': 0.1,
            'activation': "relu",
            'num_layers': 1,
            'norm': 'bn',
            'global_norm': False
        },
    )
    model(inputs)
    print(model)

    for m in model.parameters():
        m.requires_grad = False

    try:
        flops = FlopCountAnalysis(model, inputs)
        # print(flops.total())
        # print(flops.by_module_and_operator())
        print(flop_count_table(flops))
    except NameError:
        print("Warning: fvcore is not found")
