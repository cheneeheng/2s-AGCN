from typing import Optional

from torch import Tensor
from torch.nn import Module as PyTorchModule

from model.layers.torch_utils import pad_zeros
from model.architecture.sgn.blocks import Embedding


class FeatureExtractor(PyTorchModule):
    def __init__(self,
                 in_pos: int,
                 in_vel: int,
                 in_pos_emb_kwargs: dict,
                 in_vel_emb_kwargs: dict,
                 ):
        super(FeatureExtractor, self).__init__()
        self.in_pos = in_pos
        self.in_vel = in_vel
        if self.in_pos > 0:
            self.pos_embed = Embedding(**in_pos_emb_kwargs)
        if self.in_vel > 0:
            self.vel_embed = Embedding(**in_vel_emb_kwargs)
        if self.in_pos == 0 and self.in_vel == 0:
            raise ValueError("Input args are faulty...")

    def forward(self, x: Tensor) -> Optional[Tensor]:
        # x : n,c,v,t
        dif = x[:, :, :, 1:] - x[:, :, :, 0:-1]  # n,c,v,t-1
        dif = pad_zeros(dif)
        if self.in_pos > 0 and self.in_vel > 0:
            pos = self.pos_embed(x)
            vel = self.vel_embed(dif)
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
