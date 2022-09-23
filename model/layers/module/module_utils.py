# Utility functions for modules
# ##############################################################################

import torch

from typing import Any, Tuple, Type

from model.layers.torch_utils import null_fn
from model.layers.module.block import Conv
from model.layers.module.layernorm import LayerNorm


def get_normalization_fn(norm_type: str) -> Tuple[Type[torch.nn.Module],
                                                  Type[torch.nn.Module]]:
    if 'bn' in norm_type:
        return torch.nn.BatchNorm1d, torch.nn.BatchNorm2d
    elif 'ln' in norm_type:
        return LayerNorm, LayerNorm
    else:
        raise ValueError("Unknown norm_type ...")


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
