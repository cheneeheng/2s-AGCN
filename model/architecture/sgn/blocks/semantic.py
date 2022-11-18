import torch
from torch import nn
from torch import Tensor
from torch.nn import Module as PyTorchModule

from typing import Tuple, Optional, Union, Type, List

from model.layers import Module
from model.layers import Conv
from model.layers import residual as res
from model.layers import null_fn


T1 = Type[PyTorchModule]
T2 = List[Optional[Type[PyTorchModule]]]
T3 = Union[List[int], int]

EMB_MODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]


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


class OneHotTensor(PyTorchModule):
    def __init__(self, dim_eye: int, dim_length: int, mode: int):
        super(OneHotTensor, self).__init__()
        onehot = torch.eye(dim_eye, dim_eye)
        onehot = onehot.unsqueeze(0).unsqueeze(0)
        onehot = onehot.repeat(1, dim_length, 1, 1)
        if mode == 0:
            onehot = onehot.permute(0, 3, 2, 1)
        elif mode == 1:
            onehot = onehot.permute(0, 3, 1, 2)
        elif mode == 2:
            pass
        else:
            raise ValueError("Unknown mode")
        self.register_buffer("onehot", onehot)

    def forward(self, bs: int) -> Tensor:
        x = self.onehot.repeat(bs, 1, 1, 1)
        return x


class SemanticEmbedding(PyTorchModule):
    def __init__(self,
                 num_point: int,
                 num_segment: int,
                 sem_spa: int = 0,
                 sem_tem: int = 0,
                 sem_cls: int = 0,
                 sem_spa_emb_kwargs: Optional[dict] = None,
                 sem_tem_emb_kwargs: Optional[dict] = None,
                 sem_cls_emb_kwargs: Optional[dict] = None,
                 ):
        super(SemanticEmbedding, self).__init__()
        self.num_point = num_point
        self.num_segment = num_segment
        self.sem_spa = sem_spa
        self.sem_tem = sem_tem
        self.sem_cls = sem_cls
        # Joint Embedding
        if self.sem_spa > 0:
            self.spa_onehot = OneHotTensor(
                sem_spa_emb_kwargs['in_channels'], self.num_segment, mode=0)
            self.spa_embedding = Embedding(**sem_spa_emb_kwargs)
        # Frame Embedding
        if self.sem_tem > 0:
            self.tem_onehot = OneHotTensor(
                sem_tem_emb_kwargs['in_channels'], self.num_point, mode=1)
            self.tem_embedding = Embedding(**sem_tem_emb_kwargs)
        # Class Embedding
        if self.sem_cls > 0:
            self.cls_onehot = OneHotTensor(
                sem_cls_emb_kwargs['in_channels'], 1, mode=2)
            self.cls_embedding = Embedding(**sem_cls_emb_kwargs)

    def forward(self, x: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        spa_emb, tem_emb, cls_emb = None, None, None
        if self.sem_spa > 0:
            spa_emb = self.spa_embedding(self.spa_onehot(x.shape[0]))
        if self.sem_tem > 0:
            tem_emb = self.tem_embedding(self.tem_onehot(x.shape[0]))
        if self.sem_cls > 0:
            cls_emb = self.cls_embedding(self.cls_onehot(x.shape[0]))
        return spa_emb, tem_emb, cls_emb
