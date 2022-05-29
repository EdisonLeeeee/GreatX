import torch.nn as nn

from graphwar.nn.layers import SoftMedianConv, Sequential, activations
from graphwar.utils import wrapper


class SoftMedianGCN(nn.Module):
    """Graph Convolution Network (GCN) with Soft Median Aggregation

    Example
    -------
    # SoftMedianGCN with one hidden layer
    >>> model = SoftMedianGCN(100, 10)
    # SoftMedianGCN with two hidden layers
    >>> model = SoftMedianGCN(100, 10, hids=[32, 16], acts=['relu', 'elu'])
    # SoftMedianGCN with two hidden layers, without activation at the first layer
    >>> model = SoftMedianGCN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    """

    @wrapper
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hids: list = [16],
                 acts: list = ['relu'],
                 dropout: float = 0.5,
                 bn: bool = False,
                 bias: bool = True,
                 row_normalize: bool = False,
                 normalize: bool = False,
                 cached: bool = True):
        r"""
        Parameters
        ----------
        in_channels : int, 
            the input dimensions of model
        out_channels : int, 
            the output dimensions of model
        hids : list, optional
            the number of hidden units of each hidden layer, by default [16]
        acts : list, optional
            the activation function of each hidden layer, by default ['relu']
        dropout : float, optional
            the dropout ratio of model, by default 0.5
        bias : bool, optional
            whether to use bias in the layers, by default True
        bn: bool, optional
            whether to use `BatchNorm1d` after the convolution layer, by default False          
        """

        super().__init__()

        conv = []
        assert len(hids) == len(acts)
        for hid, act in zip(hids, acts):
            conv.append(SoftMedianConv(in_channels,
                                       hid,
                                       bias=bias,
                                       normalize=normalize,
                                       row_normalize=row_normalize,
                                       cached=cached))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_channels = hid
        conv.append(SoftMedianConv(in_channels, out_channels, bias=bias,
                                   normalize=normalize,
                                   row_normalize=row_normalize,
                                   cached=cached))
        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.cache_clear()

    def cache_clear(self):
        for conv in self.conv:
            if hasattr(conv, '_cached_edges'):
                conv._cached_edges = None
        return self

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)
