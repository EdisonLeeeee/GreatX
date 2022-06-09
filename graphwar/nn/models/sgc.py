import torch.nn as nn

from graphwar.utils import wrapper
from graphwar.nn.layers import SGConv, Sequential, activations


class SGC(nn.Module):
    r"""The simplified graph convolutional operator from 
    the `"Simplifying Graph Convolutional Networks"
    <https://arxiv.org/abs/1902.07153>`_ paper (ICML'19)

    Parameters
    ----------
    in_channels : int, 
        the input dimensions of model
    out_channels : int, 
        the output dimensions of model
    hids : list, optional
        the number of hidden units for each hidden layer, by default []
    acts : list, optional
        the activation function for each hidden layer, by default []
    K : int, optional
        the number of propagation steps, by default 2 
    dropout : float, optional
        the dropout ratio of model, by default 0.
    bias : bool, optional
        whether to use bias in the layers, by default True
    cached : bool, optional
        whether the layer will cache
        the computation of :math:`(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2})^K` on first execution, and will use the
        cached version for further executions, by default True            
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer, by default False     

    Note
    ----
    To accept a different graph as inputs, please call :meth:`cache_clear` first
    to clear cached results.

    It is convenient to extend the number of layers with different or the same
    hidden units (activation functions) using :meth:`graphwar.utils.wrapper`. 

    See Examples below:

    Examples
    --------
    >>> # SGC without hidden layer
    >>> model = SGC(100, 10)

    >>> # SGC with two hidden layers
    >>> model = SGC(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # SGC with two hidden layers, without activation at the first layer
    >>> model = SGC(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # SGC with very deep architectures, each layer has elu as activation function
    >>> model = SGC(100, 10, hids=[16]*8, acts=['elu'])

    See also
    --------
    :class:`graphwar.nn.layers.SGConv`        

    """

    @wrapper
    def __init__(self,
                 in_channels,
                 out_channels,
                 hids: list = [],
                 acts: list = [],
                 K: int = 2,
                 dropout: float = 0.,
                 bias: bool = True,
                 cached: bool = True,
                 bn: bool = False):
        super().__init__()

        conv = []
        for i, (hid, act) in enumerate(zip(hids, acts)):
            if i == 0:
                conv.append(SGConv(in_channels,
                                   hid,
                                   bias=bias,
                                   K=K,
                                   cached=cached))
            else:
                conv.append(nn.Linear(in_channels, hid, bias=bias))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_channels = hid

        if not hids:
            conv.append(SGConv(in_channels,
                               out_channels,
                               bias=bias,
                               K=K,
                               cached=cached))
        else:
            conv.append(nn.Linear(in_channels, out_channels, bias=bias))

        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def cache_clear(self):
        """Clear cached inputs or intermediate results."""
        for layer in self.conv:
            if hasattr(layer, 'cache_clear'):
                layer.cache_clear()
        return self

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)
