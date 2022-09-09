import math

import torch
from torch import Tensor, nn


class TensorGCNConv(nn.Module):
    r"""The rotbust tensor graph convolutional operator from 
    the `"Robust Tensor Graph Convolutional Networks 
    via T-SVD based Graph Augmentation"
    <https://dl.acm.org/doi/abs/10.1145/3534678.3539436>`_ paper (KDD'22)

    Parameters
    ----------
    in_channels : int
        dimensions of int samples
    out_channels : int
        dimensions of output samples
    num_nodes : int
        number of input nodes        
    num_channels : int
        number of input channels (adjacency matrixs)          
    bias : bool, optional
        whether to use bias in the layers, by default True    

    See also
    --------
    :class:`~greatx.nn.models.supervised.RTGCN`         
    """

    def __init__(self, in_channels: int, out_channels: int,
                 num_nodes: int, num_channels: int, bias: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels

        self.weight = nn.Parameter(torch.Tensor(in_channels,
                                                out_channels,
                                                num_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_nodes,
                                                  out_channels,
                                                  num_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: Tensor, adjs: Tensor) -> Tensor:
        """"""
        if x.ndim == 2:
            x = x.repeat(adjs.size(-1), 1, 1).permute(1, 2, 0)
        x = self.fft_product(x, self.weight)
        out = self.fft_product(adjs, x)

        if self.bias is not None:
            out += self.bias

        return out

    @staticmethod
    def fft_product(X, Y):
        X = torch.fft.fft(X)
        Y = torch.fft.fft(Y)
        Z = torch.fft.ifft(torch.einsum('ijk,jrk->irk', X, Y))
        return Z.real

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(({self.in_channels}, {self.num_channels}), '
                f'({self.out_channels}, {self.num_channels}))')


class TensorLinear(nn.Module):
    def __init__(self, in_channels: int, num_nodes: int, num_channels: int,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.Tensor(num_channels, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_nodes, in_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """"""
        out = torch.einsum('ijk,kr->ijr', x, self.weight).squeeze()
        if self.bias is not None:
            out += self.bias
        return out
