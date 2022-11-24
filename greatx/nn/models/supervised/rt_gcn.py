from typing import List

import torch.nn as nn

from greatx.nn.layers import (
    Sequential,
    TensorGCNConv,
    TensorLinear,
    activations,
)
from greatx.utils import wrapper


class RTGCN(nn.Module):
    r"""The rotbust tensor graph convolutional operator from
    the `"Robust Tensor Graph Convolutional Networks
    via T-SVD based Graph Augmentation"
    <https://dl.acm.org/doi/abs/10.1145/3534678.3539436>`_ paper (KDD'22)

    Parameters
    ----------
    in_channels : int,
        the input dimensions of model
    out_channels : int,
        the output dimensions of model
    num_nodes : int
        number of input nodes
    num_channels : int
        number of input channels (adjacency matrixs)
    hids : List[int], optional
        the number of hidden units for each hidden layer, by default [16]
    acts : List[str], optional
        the activation function for each hidden layer, by default ['relu']
    dropout : float, optional
        the dropout ratio of model, by default 0.5
    bias : bool, optional
        whether to use bias in the layers, by default True
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer,
        by default False

    Examples
    --------
    >>> # RTGCN with one hidden layer
    >>> num_nodes = 2485
    >>> num_channels = 3
    >>> model = RTGCN(100, 10, num_nodes, num_channels)

    See also
    --------
    :class:`greatx.nn.layers.TensorGCNConv`

    """
    @wrapper
    def __init__(self, in_channels: int, out_channels: int, num_nodes: int,
                 num_channels: int, hids: List[int] = [16],
                 acts: List[str] = ['relu'], dropout: float = 0.5,
                 bias: bool = True, bn: bool = False):

        super().__init__()

        conv = []
        assert len(hids) == len(acts)
        for hid, act in zip(hids, acts):
            conv.append(
                TensorGCNConv(in_channels, hid, num_nodes=num_nodes,
                              num_channels=num_channels, bias=bias))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_channels = hid
        conv.append(
            TensorGCNConv(in_channels, out_channels, num_nodes=num_nodes,
                          num_channels=num_channels, bias=bias))
        conv.append(
            TensorLinear(out_channels, num_nodes=num_nodes,
                         num_channels=num_channels, bias=bias))
        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, adjs):
        """"""
        return self.conv(x, adjs)
