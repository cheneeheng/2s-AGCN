import torch

from model.module.pytorch_module_wrapper import Module, Conv
from model.torch_utils import *


class Residual(Module):
    def __init__(self,
                 layer: function,
                 mode: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 padding: int = 0,
                 dilation: Union[int, list] = 1,
                 bias: int = 0,
                 deterministic: Optional[bool] = None,
                 dropout: OTPM = None,
                 activation: OTPM = None,
                 normalization: OTPM = None,
                 prenorm: bool = False):
        super(Residual, self).__init__()
        self.layer = layer
        if mode == 0:
            self.skip = null_fn
        elif mode == 1:
            if in_ch == out_ch:
                self.skip = torch.nn.Identity()
            else:
                self.skip = Conv(in_ch, out_ch, bias=bias)
        else:
            raise ValueError("Unknown residual modes...")

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def residual_layer(residual: int, in_ch: int, out_ch: int, bias: int = 0):
    if residual == 0:
        return null_fn
    elif residual == 1:
        if in_ch == out_ch:
            return nn.Identity()
        else:
            return Conv(in_ch, out_ch, bias=bias)
    else:
        raise ValueError("Unknown residual modes...")
