from typing import List

import torch.nn as nn

from greatx.nn.layers import MedianConv, Sequential, activations
from greatx.utils import wrapper


class MedianGCN(nn.Module):
    r"""Graph Convolution Network (GCN) with
    median aggregation (MedianGCN)
    from the `"Understanding Structural Vulnerability
    in Graph Convolutional Networks"
    <https://www.ijcai.org/proceedings/2021/310>`_ paper (IJCAI'21)

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
    reduce : str
        aggregation function, including {'median', 'sample_median'},
        where :obj:`median` uses the exact median as the aggregation function,
        while :obj:`sample_median` appropriates the median with a fixed set
        of sampled nodes. :obj:`sample_median` is much faster and
        more scalable than :obj:`median`. By default, :obj:`median` is used.
    dropout : float, optional
        the dropout ratio of model, by default 0.5
    bias : bool, optional
        whether to use bias in the layers, by default True
    normalize : bool, optional
        whether to compute symmetric normalization
        coefficients on the fly, by default False
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer,
        by default False

    Examples
    --------
    >>> # MedianGCN with one hidden layer
    >>> model = MedianGCN(100, 10)

    >>> # MedianGCN with two hidden layers
    >>> model = MedianGCN(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # MedianGCN with two hidden layers, without first activation
    >>> model = MedianGCN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # MedianGCN with deep architectures, each layer has elu activation
    >>> model = MedianGCN(100, 10, hids=[16]*8, acts=['elu'])

    >>> # MedianGCN with sample median aggregation
    >>> model = MedianGCN(100, 10, reduce='sample_median')

    See also
    --------
    :class:`greatx.nn.layers.MedianConv`

    """
    @wrapper
    def __init__(self, in_channels: int, out_channels: int,
                 hids: List[int] = [16], acts: List[str] = ['relu'],
                 reduce: str = 'median', dropout: float = 0.5,
                 bn: bool = False, normalize: bool = False, bias: bool = True):

        super().__init__()

        conv = []
        assert len(hids) == len(acts)
        for hid, act in zip(hids, acts):
            conv.append(
                MedianConv(in_channels, hid, bias=bias, normalize=normalize,
                           reduce=reduce))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_channels = hid
        conv.append(
            MedianConv(in_channels, out_channels, bias=bias,
                       normalize=normalize, reduce=reduce))
        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        return self.conv(x, edge_index, edge_weight)
