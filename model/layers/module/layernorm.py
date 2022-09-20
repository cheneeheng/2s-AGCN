from torch import nn


# This same as nn.LayerNorm, but takes N,C,* as input and output
class LayerNorm(nn.GroupNorm):
    def __init__(self, num_channels: int, **kwargs):
        kwargs['num_groups'] = 1
        kwargs['num_channels'] = num_channels
        super(LayerNorm, self).__init__(**kwargs)
