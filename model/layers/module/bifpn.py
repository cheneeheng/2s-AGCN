# BASED ON:
# https://github.com/tristandb/EfficientDet-PyTorch/blob/master/bifpn.py
# ##############################################################################

import torch
import torch.nn as nn

from typing import Union

from model.layers.module.block import Conv


class BiFPNBlock(nn.Module):
    """Bi-directional Feature Pyramid Network """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 td_kernel_size: int = 1,
                 out_kernel_size: int = 1,
                 epsilon: float = 0.0001):
        super(BiFPNBlock, self).__init__()
        assert in_channels == out_channels

        self.epsilon = epsilon

        self.p1_td = Conv(in_channels,
                          out_channels,
                          td_kernel_size,
                          td_kernel_size//2)
        self.p2_td = Conv(in_channels,
                          out_channels,
                          td_kernel_size,
                          td_kernel_size//2)
        # self.p3_td = Conv(in_channels, out_channels, td_kernel_size)

        # self.p1_out = Conv(in_channels, out_channels, out_kernel_size)
        self.p2_out = Conv(in_channels,
                           out_channels,
                           out_kernel_size,
                           out_kernel_size//2)
        self.p3_out = Conv(in_channels,
                           out_channels,
                           out_kernel_size,
                           out_kernel_size//2)

        self.w1_p1 = nn.Parameter(torch.ones(2))
        self.w1_p2 = nn.Parameter(torch.ones(2))
        self.w1_relu = nn.ReLU()
        self.w2_p2 = nn.Parameter(torch.ones(3))
        self.w2_p3 = nn.Parameter(torch.ones(2))
        self.w2_relu = nn.ReLU()

    def forward(self, x: Union[list, tuple]) -> list:
        p1_x, p2_x, p3_x = x

        # Calculate Top-Down Pathway
        w1_p1 = self.w1_relu(self.w1_p1)
        w1_p1 /= torch.sum(w1_p1, dim=0) + self.epsilon
        w1_p2 = self.w1_relu(self.w1_p2)
        w1_p2 /= torch.sum(w1_p2, dim=0) + self.epsilon
        w2_p2 = self.w2_relu(self.w2_p2)
        w2_p2 /= torch.sum(w2_p2, dim=0) + self.epsilon
        w2_p3 = self.w2_relu(self.w2_p3)
        w2_p3 /= torch.sum(w2_p3, dim=0) + self.epsilon

        p3_td = p3_x
        p2_td = self.p2_td(w1_p2[0] * p2_x + w1_p2[1] * p3_td)
        p1_td = self.p1_td(w1_p1[0] * p1_x + w1_p1[1] * p2_td)

        # Calculate Bottom-Up Pathway
        p1_out = p1_td
        p2_out = self.p2_out(w2_p2[0] * p2_x +
                             w2_p2[1] * p2_td + w2_p2[2] * p1_out)
        p3_out = self.p3_out(w2_p3[0] * p3_td + w2_p3[1] * p2_out)

        return [p1_out, p2_out, p3_out]


class BiFPN(nn.Module):
    def __init__(self,
                 in_channels: Union[list, tuple],
                 out_channels: int = 64,
                 proj_kernel_size: int = 1,
                 td_kernel_size: int = 1,
                 out_kernel_size: int = 1,
                 num_layers: int = 2,
                 epsilon: float = 0.0001):
        super(BiFPN, self).__init__()

        self.num_layers = num_layers

        self.p1 = Conv(in_channels[0],
                       out_channels,
                       proj_kernel_size,
                       proj_kernel_size//2)
        self.p2 = Conv(in_channels[1],
                       out_channels,
                       proj_kernel_size,
                       proj_kernel_size//2)
        self.p3 = Conv(in_channels[2],
                       out_channels,
                       proj_kernel_size,
                       proj_kernel_size//2)

        for i in range(self.num_layers):
            setattr(self, f'bifpn{i+1}', BiFPNBlock(out_channels,
                                                    out_channels,
                                                    td_kernel_size,
                                                    out_kernel_size,
                                                    epsilon))

    def forward(self, x: Union[list, tuple]) -> list:
        c1, c2, c3 = x
        p1_x = self.p1(c1)
        p2_x = self.p2(c2)
        p3_x = self.p3(c3)
        x = [p1_x, p2_x, p3_x]
        for i in range(self.num_layers):
            x = getattr(self, f'bifpn{i+1}')(x)
        return x


# class DepthwiseConvBlock(nn.Module):
#     """Depthwise seperable convolution. """

#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
#                  padding=0, dilation=1, freeze_bn=False):
#         super(DepthwiseConvBlock, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
#                                    stride,
#                                    padding, dilation, groups=in_channels,
#                                    bias=False)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
#                                    stride=1, padding=0, dilation=1, groups=1,
#                                    bias=False)

#         self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
#         self.act = nn.ReLU()

#     def forward(self, inputs):
#         x = self.depthwise(inputs)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return self.act(x)
