import torch
import torch.nn as nn

from graphwar.nn.layers import TAGConv, Sequential, activations
from graphwar.utils import wrapper


class TAGCN(nn.Module):
    r"""Topological adaptive graph convolution network 
    (TAGCN) from the `"Topological Adaptive Graph 
    Convolutional Networks"
    <https://arxiv.org/abs/1806.03536>`_ paper (arXiv'17)

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
    K : int
        the number of propagation steps, by default 2             
    dropout : float, optional
        the dropout ratio of model, by default 0.5
    bias : bool, optional
        whether to use bias in the layers, by default True
    normalize : bool, optional
        whether to compute symmetric normalization
        coefficients on the fly, by default False            
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer, by default False         

    Note
    ----
    It is convenient to extend the number of layers with different or the same
    hidden units (activation functions) using :meth:`graphwar.utils.wrapper`. 

    See Examples below:

    Examples
    --------
    >>> # TAGCN with one hidden layer
    >>> model = TAGCN(100, 10)

    >>> # TAGCN with two hidden layers
    >>> model = TAGCN(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # TAGCN with two hidden layers, without activation at the first layer
    >>> model = TAGCN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # TAGCN with very deep architectures, each layer has elu as activation function
    >>> model = TAGCN(100, 10, hids=[16]*8, acts=['elu'])

    See also
    --------
    :class:`graphwar.nn.layers.TAGCNConv`    

    """

    @wrapper
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hids: list = [16],
                 acts: list = ['relu'],
                 K: int = 2,
                 dropout: float = 0.5,
                 bias: bool = True,
                 normalize: bool = True,
                 bn: bool = False):

        super().__init__()

        conv = []
        assert len(hids) == len(acts)
        for hid, act in zip(hids, acts):
            conv.append(TAGConv(in_channels,
                                hid,
                                K=K,
                                bias=bias,
                                normalize=normalize))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_channels = hid
        conv.append(TAGConv(in_channels, out_channels,
                            K=K,
                            bias=bias,
                            normalize=normalize))
        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)
