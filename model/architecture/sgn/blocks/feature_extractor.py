from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module as PyTorchModule

from model.layers.torch_utils import pad_zeros
from model.architecture.sgn.blocks import Embedding


SEGMENTS = [[2, 3],
            [0, 1, 20],
            [4, 5, 6],
            [8, 9, 10],
            [16, 17, 18, 19],
            [12, 13, 14, 15],
            [7, 21, 22],
            [11, 23, 24]]

PERIOD = 4


def embedding_per_segment(x: Tensor, fn_list: torch.nn.ModuleList):
    n, _, _, t = x.shape
    x_segs = []
    for idx, s in enumerate(SEGMENTS):
        x_s = torch.stack([x[:, :, i, :] for i in s], 2)
        x_s = torch.stack([x_s[:, :, :, i*PERIOD:(i+1)*PERIOD]
                           for i in range(t//PERIOD)], 3)
        x_s = x_s.permute((0, 4, 2, 1, 3))
        x_s = x_s.reshape(n, -1, 1, x_s.shape[-1])
        x_segs.append(fn_list[idx](x_s))
    return torch.cat(x_segs, 2)  # n,c',v',t'


class FeatureExtractor(PyTorchModule):
    def __init__(self,
                 in_pos: int,
                 in_vel: int,
                 in_pos_emb_kwargs: dict,
                 in_vel_emb_kwargs: dict,
                 fusion: int = 1
                 ):
        super(FeatureExtractor, self).__init__()

        self.fusion = fusion
        self.in_pos = in_pos
        self.in_vel = in_vel

        if self.in_pos > 1000:
            self.pos_embed = torch.nn.ModuleList()
            for s in SEGMENTS:
                kwargs = in_pos_emb_kwargs.copy()
                kwargs['in_channels'] = kwargs['in_channels']*len(s)*4
                kwargs['mode'] = kwargs['mode'] % 1000
                kwargs['num_point'] = 1
                self.pos_embed.append(Embedding(**kwargs))
        elif self.in_pos > 0:
            self.pos_embed = Embedding(**in_pos_emb_kwargs)

        if self.in_vel > 1000:
            self.vel_embed = torch.nn.ModuleList()
            for s in SEGMENTS:
                kwargs = in_vel_emb_kwargs.copy()
                kwargs['in_channels'] = kwargs['in_channels']*len(s)*4
                kwargs['mode'] = kwargs['mode'] % 1000
                kwargs['num_point'] = 1
                self.vel_embed.append(Embedding(**kwargs))
        elif self.in_vel > 0:
            self.vel_embed = Embedding(**in_vel_emb_kwargs)

        if self.in_pos == 0 and self.in_vel == 0:
            raise ValueError("Input args are faulty...")

    def forward(self, x: Tensor) -> Optional[Tensor]:
        n, c, v, t = x.shape
        dif = x[:, :, :, 1:] - x[:, :, :, 0:-1]  # n,c,v,t-1
        dif = pad_zeros(dif)

        if self.in_pos > 0 and self.in_vel > 0:
            if self.in_pos > 1000 and self.in_vel > 1000:
                pos = embedding_per_segment(x, self.pos_embed)
                vel = embedding_per_segment(x, self.vel_embed)
            else:
                pos = self.pos_embed(x)
                vel = self.vel_embed(dif)

            if self.fusion == 0:
                dy1 = torch.cat([pos, vel], dim=1)  # n,c,v,t
            elif self.fusion == 1:
                dy1 = pos + vel  # n,c,v,t
            return dy1, pos, vel

        elif self.in_pos > 0:
            dy1 = self.pos_embed(x)
            return dy1, dy1, None

        elif self.in_vel > 0:
            dy1 = self.vel_embed(dif)
            return dy1, None, dy1

        else:
            return None, None, None
