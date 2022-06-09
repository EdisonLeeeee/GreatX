import torch.nn as nn

from graphwar.nn.layers import SoftMedianConv, Sequential, activations
from graphwar.utils import wrapper


class SoftMedianGCN(nn.Module):
    r"""Graph Convolution Network (GCN) with 
    soft median aggregation (MedianGCN)
    from the `"Robustness of Graph Neural Networks 
    at Scale" <https://arxiv.org/abs/2110.14038>`_ paper 
    (NeurIPS'21)

    Parameters
    ----------
    in_channels : int, 
        the input dimensions of model
    out_channels : int, 
        the output dimensions of model
    hids : list, optional
        the number of hidden units for each hidden layer, by default [16]
    acts : list, optional
        the activation function for each hidden layer, by default ['relu']
    dropout : float, optional
        the dropout ratio of model, by default 0.5
    bias : bool, optional
        whether to use bias in the layers, by default True
    normalize : bool, optional
        whether to compute symmetric normalization
        coefficients on the fly, by default False             
    row_normalize : bool, optional
        whether to perform row-normalization on the fly, by default True           
    cached : bool, optional
        whether the layer will cache
        the computation of :math:`(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2})` and sorted edges on first execution, 
        and will use the cached version for further executions, by default False
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer, by default False   

    Note
    ----
    It is convenient to extend the number of layers with different or the same
    hidden units (activation functions) using :meth:`graphwar.utils.wrapper`. 

    See Examples below:

    Examples
    --------
    >>> # SoftMedianGCN with one hidden layer
    >>> model = SoftMedianGCN(100, 10)

    >>> # SoftMedianGCN with two hidden layers
    >>> model = SoftMedianGCN(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # SoftMedianGCN with two hidden layers, without activation at the first layer
    >>> model = SoftMedianGCN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # SoftMedianGCN with very deep architectures, each layer has elu as activation function
    >>> model = SoftMedianGCN(100, 10, hids=[16]*8, acts=['elu'])

    See also
    --------
    :class:`graphwar.nn.layers.SoftMedianConv`    

    """

    @wrapper
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hids: list = [16],
                 acts: list = ['relu'],
                 dropout: float = 0.5,
                 bias: bool = True,
                 normalize: bool = False,
                 row_normalize: bool = False,
                 cached: bool = True,
                 bn: bool = False):

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
        """Clear cached inputs or intermediate results."""
        for conv in self.conv:
            if hasattr(conv, '_cached_edges'):
                conv._cached_edges = None
        return self

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)
