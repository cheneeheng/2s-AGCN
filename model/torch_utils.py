from typing import Any

import torch


__all__ = ['null_fn', 'init_zeros', 'pad_zeros']


def null_fn(x: Any) -> int:
    return 0


def init_zeros(x: torch.Tensor):
    return torch.nn.init.constant_(x, 0)


def pad_zeros(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.new(*x.shape[:-1], 1).zero_(), x], dim=-1)
