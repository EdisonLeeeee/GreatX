from typing import List

import torch.nn as nn

from greatx.nn.layers import GCNConv, Sequential, activations
from greatx.utils import wrapper


class GCN(nn.Module):
    r"""Graph Convolution Network (GCN) from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper (ICLR'17)

    Parameters
    ----------
    in_channels : int,
        the input dimensions of model
    out_channels : int,
        the output dimensions of model
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
    normalize : bool, optional
        whether to compute symmetric normalization
        coefficients on the fly, by default True

    Examples
    --------
    >>> # GCN with one hidden layer
    >>> model = GCN(100, 10)

    >>> # GCN with two hidden layers
    >>> model = GCN(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # GCN with two hidden layers, without first activation
    >>> model = GCN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # GCN with deep architectures, each layer has elu activation
    >>> model = GCN(100, 10, hids=[16]*8, acts=['elu'])

    See also
    --------
    :class:`greatx.nn.layers.GCNConv`

    """
    @wrapper
    def __init__(self, in_channels: int, out_channels: int,
                 hids: List[int] = [16], acts: List[str] = ['relu'],
                 dropout: float = 0.5, bias: bool = True, bn: bool = False,
                 normalize: bool = True):

        super().__init__()

        conv = []
        assert len(hids) == len(acts)
        for hid, act in zip(hids, acts):
            conv.append(
                GCNConv(in_channels, hid, bias=bias, normalize=normalize))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_channels = hid
        conv.append(
            GCNConv(in_channels, out_channels, bias=bias, normalize=normalize))
        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        return self.conv(x, edge_index, edge_weight)
