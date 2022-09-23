# Utility functions for torch library.
# ##############################################################################

import torch

from typing import Any, Tuple, Type


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


def tensor_list_sum(x: list) -> torch.Tensor:
    return torch.sum(torch.stack(x, dim=0), dim=0)


def tensor_list_mean(x: list) -> torch.Tensor:
    return torch.mean(torch.stack(x, dim=0), dim=0)


def fuse_features(x1: torch.Tensor,
                  x2: torch.Tensor,
                  mode: int) -> torch.Tensor:
    if mode == 0:
        return torch.cat([x1, x2], 1)
    elif mode == 1:
        return x1 + x2
    else:
        raise ValueError('Unknown feature fusion arg')
