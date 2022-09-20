# Based on:
# https://github.com/thuml/Autoformer/blob/main/layers/Autoformer_EncDec.py

import torch


class MovingAverage(torch.nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size: int, stride: int):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool2d(kernel_size=(1, kernel_size),
                                      stride=(1, stride),
                                      padding=(0, 0))

    def forward(self, x: torch.Tensor):
        # padding on the both ends of time series
        front = x[:, :, :, 0:1].repeat(1, 1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, :, -1:].repeat(1, 1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x


class SeriesDecomposition(torch.nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size: int):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean  # seasonal, trend
