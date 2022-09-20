# Utility functions for torch library.
# ##############################################################################

import torch

from typing import Any, Tuple, Type

from model.layers.module.core import Conv
from model.layers.module.layernorm import LayerNorm


__all__ = ['null_fn', 'init_zeros', 'pad_zeros',
           'get_activation_fn', 'get_normalization_fn',
           'tensor_list_sum', 'tensor_list_mean',
           'residual', 'fuse_features']


def null_fn(x: Any) -> torch.Tensor:
    return 0


def init_zeros(x: torch.Tensor):
    return torch.nn.init.constant_(x, 0)


def pad_zeros(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.new(*x.shape[:-1], 1).zero_(), x], dim=-1)


def get_activation_fn(act_type: str) -> Type[Type[torch.nn.Module]]:
    if act_type == 'relu':
        return torch.nn.ReLU
    elif act_type == 'gelu':
        return torch.nn.GELU
    else:
        raise ValueError("Unknown act_type ...")


def get_normalization_fn(norm_type: str) -> Tuple[Type[torch.nn.Module],
                                                  Type[torch.nn.Module]]:
    if 'bn' in norm_type:
        return torch.nn.BatchNorm1d, torch.nn.BatchNorm2d
    elif 'ln' in norm_type:
        return LayerNorm, LayerNorm
    else:
        raise ValueError("Unknown norm_type ...")


def tensor_list_sum(x: list) -> torch.Tensor:
    return torch.sum(torch.stack(x, dim=0), dim=0)


def tensor_list_mean(x: list) -> torch.Tensor:
    return torch.mean(torch.stack(x, dim=0), dim=0)


def residual(mode: int, in_ch: int, out_ch: int, bias: int = 0):
    if mode == 0:
        return null_fn
    elif mode == 1:
        if in_ch == out_ch:
            return torch.nn.Identity()
        else:
            return Conv(in_ch, out_ch, bias=bias)
    else:
        raise ValueError("Unknown residual modes...")


def fuse_features(x1: torch.Tensor,
                  x2: torch.Tensor,
                  mode: int) -> torch.Tensor:
    if mode == 0:
        return torch.cat([x1, x2], 1)
    elif mode == 1:
        return x1 + x2
    else:
        raise ValueError('Unknown feature fusion arg')
