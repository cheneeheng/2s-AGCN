from multiprocessing import pool
import torch
from torch import nn
from torch import Tensor
from torch.nn import Module as PyTorchModule

from typing import Tuple, Optional, Union, Type, List, Any
from model.layers.torch_utils import null_fn

from model.resource.common_ntu import *
from model.layers import Module
from model.layers import Conv
from model.layers import Pool
from model.layers import ASPP
from model.layers import residual as res
from model.layers import Transformer
from model.layers import SeriesDecomposition
from model.layers import PositionalEncoding
from model.layers import CosSinPositionalEncoding

T1 = Type[PyTorchModule]
T2 = List[Optional[Type[PyTorchModule]]]
T3 = Union[List[int], int]


class MHATemporal(PyTorchModule):
    def __init__(self,
                 num_layers: int,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float,
                 activation: str,
                 norm=None,
                 d_head=None,
                 dim_feedforward_output=None,
                 global_norm=None,
                 pos_enc=None,
                 max_len=20,
                 **kwargs):
        super(MHATemporal, self).__init__()
        if norm is not None:
            assert d_head is not None
            assert dim_feedforward_output is not None
            assert global_norm is not None
            self.transformer = Transformer(
                dim=d_model,
                depth=num_layers,
                heads=nhead,
                dim_head=d_head,
                dropout=dropout,
                mlp_dim=dim_feedforward,
                mlp_out_dim=dim_feedforward_output,
                activation=activation,
                norm=norm,
                global_norm=global_norm,
                **kwargs
                # v_proj=kwargs.get('v_proj', True),
                # res_proj=kwargs.get('res_proj', False),
                # output_dim=kwargs.get('d_out', d_model)
            )
        else:
            self.num_layers = num_layers
            for i in range(self.num_layers):
                setattr(self,
                        f'layer{i+1}',
                        nn.TransformerEncoderLayer(
                            d_model=d_model,
                            nhead=nhead,
                            dim_feedforward=dim_feedforward,
                            dropout=dropout,
                            activation=activation,
                            layer_norm_eps=1e-5,
                            batch_first=True,
                        ))
        if pos_enc is None:
            self.pos_enc = nn.Identity()
        elif pos_enc == 'abs':
            self.pos_enc = PositionalEncoding(d_model[0], max_len=max_len)
        elif pos_enc == 'cos':
            self.pos_enc = CosSinPositionalEncoding(
                d_model[0], max_len=max_len)
        else:
            raise ValueError("unknown pos_enc")

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[list]]:
        x = self.pos_enc(x)
        if hasattr(self, 'transformer'):
            x, attn_list = self.transformer(x)
        else:
            attn_list = None
            for i in range(self.num_layers):
                x = getattr(self, f'layer{i+1}')(x)
        return x, attn_list


class MLPTemporal(PyTorchModule):
    def __init__(self,
                 channels: List[int],
                 kernel_sizes: List[int] = [3, 1],
                 paddings: List[int] = [1, 0],
                 dilations: List[int] = [1, 1],
                 biases: List[int] = [0, 0],
                 residuals: List[int] = [0, 0],
                 dropouts: T2 = [nn.Dropout2d, None],
                 activations: T2 = [nn.ReLU, nn.ReLU],
                 normalizations: T2 = [nn.BatchNorm2d, nn.BatchNorm2d],
                 maxpool_kwargs: Optional[dict] = None,
                 residual: int = 0,
                 prenorm: bool = False
                 ):
        super(MLPTemporal, self).__init__()

        if maxpool_kwargs is not None:
            self.pool = nn.MaxPool2d(**maxpool_kwargs)
        else:
            self.pool = nn.Identity()

        self.res = res(residual, channels[0], channels[-1], biases[0])

        self.num_layers = len(channels) - 1
        for i in range(self.num_layers):

            if normalizations[i] is None:
                norm_func = None
            else:
                if prenorm:
                    def norm_func(): return normalizations[i](channels[i])
                else:
                    def norm_func(): return normalizations[i](channels[i+1])

            setattr(self,
                    f'cnn{i+1}',
                    Conv(channels[i],
                         channels[i+1],
                         kernel_size=kernel_sizes[i],
                         padding=paddings[i],
                         dilation=dilations[i],
                         bias=biases[i],
                         activation=activations[i],
                         normalization=norm_func,
                         dropout=dropouts[i],
                         prenorm=prenorm,
                         deterministic=False if dilations[i] > 1 else True)
                    )

            setattr(self,
                    f'res{i+1}',
                    res(residuals[i], channels[i], channels[i+1], biases[i]))

    def forward(self, x: Tensor, x_n: Optional[Tensor] = None) -> Tensor:
        # x: n,c,v,t ; v=1 due to SMP
        x0 = x if x_n is None else x_n
        x = self.pool(x)
        for i in range(self.num_layers):
            x = getattr(self, f'cnn{i+1}')(x) + getattr(self, f'res{i+1}')(x)
        x += self.res(x0)
        return x


class MLPTemporalDecompose(PyTorchModule):
    def __init__(self,
                 channels: List[int],
                 kernel_sizes: List[int] = [3, 1],
                 paddings: List[int] = [1, 0],
                 dilations: List[int] = [1, 1],
                 biases: List[int] = [0, 0],
                 residuals: List[int] = [0, 0],
                 dropouts: T2 = [nn.Dropout2d, None],
                 activations: T2 = [nn.ReLU, nn.ReLU],
                 normalizations: T2 = [nn.BatchNorm2d, nn.BatchNorm2d],
                 maxpool_kwargs: Optional[dict] = None,
                 residual: int = 0,
                 prenorm: bool = False,
                 decomp_kernel_size: int = 3
                 ):
        super(MLPTemporalDecompose, self).__init__()
        kwargs = {'channels': channels,
                  'kernel_sizes': kernel_sizes,
                  'paddings': paddings,
                  'dilations': dilations,
                  'biases': biases,
                  'residuals': residuals,
                  'dropouts': dropouts,
                  'activations': activations,
                  'normalizations': normalizations,
                  'maxpool_kwargs': maxpool_kwargs,
                  'residual': residual,
                  'prenorm': prenorm
                  }
        self.cnn_raw = MLPTemporal(**kwargs)
        self.cnn_season = MLPTemporal(**kwargs)
        self.cnn_trend = MLPTemporal(**kwargs)
        self.decomp = SeriesDecomposition(decomp_kernel_size)

    def forward(self, x: Tensor) -> List[Tensor]:
        # x: n,c,v,t ; v=1 due to SMP
        x_se, x_tr = self.decomp(x)
        x1 = self.cnn_raw(x)
        x2 = self.cnn_season(x_se)
        x3 = self.cnn_trend(x_tr)
        return [x1, x2, x3]


class MLPTemporalPool(PyTorchModule):
    def __init__(self,
                 channels: List[int],
                 kernel_sizes: List[int] = [3, 1],
                 paddings: List[int] = [1, 0],
                 dilations: List[int] = [1, 1],
                 biases: List[int] = [0, 0],
                 residuals: List[int] = [0, 0],
                 dropouts: T2 = [nn.Dropout2d, None],
                 activations: T2 = [nn.ReLU, nn.ReLU],
                 normalizations: T2 = [nn.BatchNorm2d, nn.BatchNorm2d],
                 maxpool_kwargs: Optional[dict] = None,
                 residual: int = 0,
                 prenorm: bool = False,
                 pool_kernel_sizes: List[int] = [3, 5, 7, 9],
                 ):
        super(MLPTemporalPool, self).__init__()
        kwargs = {'channels': channels[-2:],
                  'kernel_sizes': kernel_sizes[-1:],
                  'paddings': paddings[-1:],
                  'dilations': dilations[-1:],
                  'biases': biases[-1:],
                  'residuals': residuals[-1:],
                  'dropouts': dropouts[-1:],
                  'activations': activations[-1:],
                  'normalizations': normalizations[-1:],
                  'maxpool_kwargs': maxpool_kwargs,
                  'residual': residual,
                  'prenorm': prenorm
                  }

        self.pool_len = len(pool_kernel_sizes)

        if normalizations[0] is None:
            _normalization = None
        else:
            def _normalization(): return normalizations[0](channels[1])

        for i, k in enumerate(pool_kernel_sizes):
            setattr(self, f'pad{i+1}',
                    nn.ReplicationPad2d(((k - 1) // 2, (k - 1) // 2, 0, 0)))
            setattr(self, f'pool{i+1}',
                    Pool(channels[0],
                         channels[1],
                         pooling=nn.AvgPool2d((1, k),
                                              stride=(1, 1),
                                              padding=(0, 0)),
                         kernel_size=kernel_sizes[0],
                         padding=paddings[0],
                         dilation=dilations[0],
                         bias=biases[0],
                         dropout=dropouts[0],
                         activation=activations[0],
                         normalization=_normalization))
            setattr(self, f'cnn{i+1}', MLPTemporal(**kwargs))

    def forward(self, x: Tensor) -> List[Tensor]:
        # x: n,c,v,t ; v=1 due to SMP
        output = []
        for i in range(self.pool_len):
            x1 = getattr(self, f'pad{i+1}')(x)
            x2 = getattr(self, f'pool{i+1}')(x1)
            x3 = getattr(self, f'cnn{i+1}')(x2)
            output.append(x3)
        return output


class TemporalBranch(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,  # t_kernel
                 bias: int = 0,
                 dropout: Optional[Type[PyTorchModule]] = None,
                 activation: Optional[Type[PyTorchModule]] = None,
                 normalization: Optional[Type[PyTorchModule]] = None,
                 prenorm: bool = False,
                 t_mode: int = 0,
                 aspp_rates: Optional[List[int]] = None,
                 maxpool_kwargs: Optional[dict] = None,
                 mha_kwargs: Optional[dict] = None,
                 decomp_kernel_size: int = 3,
                 pool_kernel_sizes: List[int] = [3, 5, 7, 9],
                 ):
        super(TemporalBranch, self).__init__(in_channels,
                                             out_channels,
                                             kernel_size=kernel_size,
                                             bias=bias,
                                             dropout=dropout,
                                             activation=activation,
                                             normalization=normalization,
                                             prenorm=prenorm)

        # aspp
        if aspp_rates is None or len(aspp_rates) == 0:
            self.aspp = nn.Identity()
        else:
            self.aspp = ASPP(self.in_channels,
                             self.in_channels,
                             bias=self.bias,
                             dilation=aspp_rates,
                             dropout=self.dropout,
                             activation=self.activation,
                             normalization=self.normalization)

        # Temporal branch ------------------------------------------------------
        self.t_mode = t_mode

        if t_mode == 0:
            self.cnn = nn.Identity()
        elif t_mode == 1:
            idx = 2
            self.cnn = MLPTemporal(
                channels=[self.in_channels, self.in_channels,
                          self.out_channels],
                kernel_sizes=[self.kernel_size, 1],
                paddings=[self.kernel_size//2, 0],
                biases=[self.bias for _ in range(idx)],
                residuals=[0 for _ in range(idx)],
                dropouts=[self.dropout, None],
                activations=[self.activation for _ in range(idx)],
                normalizations=[self.normalization for _ in range(idx)],
                maxpool_kwargs=maxpool_kwargs,
                prenorm=self.prenorm
            )
        elif t_mode == 2:
            idx = 2
            self.cnn = MLPTemporal(
                channels=[self.in_channels, self.in_channels,
                          self.out_channels],
                kernel_sizes=[self.kernel_size, 1],
                paddings=[self.kernel_size//2, 0],
                biases=[self.bias for _ in range(idx)],
                residuals=[1 for _ in range(idx)],
                dropouts=[self.dropout, None],
                activations=[self.activation for _ in range(idx)],
                normalizations=[self.normalization for _ in range(idx)],
                maxpool_kwargs=maxpool_kwargs,
                prenorm=self.prenorm
            )
        elif t_mode == 3:
            self.cnn = MHATemporal(**mha_kwargs)
        elif t_mode == 4:
            idx = 2
            self.cnn = MLPTemporalDecompose(
                channels=[self.in_channels, self.in_channels,
                          self.out_channels],
                kernel_sizes=[self.kernel_size, 1],
                paddings=[self.kernel_size//2, 0],
                biases=[self.bias for _ in range(idx)],
                residuals=[1 for _ in range(idx)],
                dropouts=[self.dropout, None],
                activations=[self.activation for _ in range(idx)],
                normalizations=[self.normalization for _ in range(idx)],
                maxpool_kwargs=maxpool_kwargs,
                prenorm=self.prenorm,
                decomp_kernel_size=decomp_kernel_size
            )
        elif t_mode == 5:
            idx = 2
            self.cnn = MLPTemporalPool(
                channels=[self.in_channels, self.in_channels,
                          self.out_channels],
                kernel_sizes=[self.kernel_size, 1],
                paddings=[self.kernel_size//2, 0],
                biases=[self.bias for _ in range(idx)],
                residuals=[1 for _ in range(idx)],
                dropouts=[self.dropout, None],
                activations=[self.activation for _ in range(idx)],
                normalizations=[self.normalization for _ in range(idx)],
                maxpool_kwargs=maxpool_kwargs,
                prenorm=self.prenorm,
                pool_kernel_sizes=pool_kernel_sizes
            )
        else:
            raise ValueError('Unknown t_mode')

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[list]]:
        attn_list = None
        N, _, _, T = x.shape
        x = self.aspp(x)
        if self.t_mode == 3:
            x = x.permute(0, 3, 2, 1).contiguous()  # n,t,v,c
            x = x.reshape(N, T, -1)
            x, attn_list = self.cnn(x)
            x = x.reshape(N, T, 1, -1)
            x = x.permute(0, 3, 2, 1).contiguous()  # n,c,v,t
        else:
            x, attn_list = self.cnn(x), None
        return x, attn_list
