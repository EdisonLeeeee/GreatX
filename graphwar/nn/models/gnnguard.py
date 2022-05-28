import torch
import torch.nn as nn

from graphwar.nn.layers import GCNConv, Sequential, activations
from graphwar.defense import GNNGUARD as GNNGUARDLayer
from graphwar.utils import wrapper

class GNNGUARD(nn.Module):
    """Graph Convolution Network (GCN) with GNNGUARD

    Example
    -------
    # GNNGUARD with one hidden layer
    >>> model = GNNGUARD(100, 10)
    # GNNGUARD with two hidden layers
    >>> model = GNNGUARD(100, 10, hids=[32, 16], acts=['relu', 'elu'])
    # GNNGUARD with two hidden layers, without activation at the first layer
    >>> model = GNNGUARD(100, 10, hids=[32, 16], acts=[None, 'relu'])

    """

    @wrapper
    def __init__(self,
                 in_feats: int,
                 out_feats: int,
                 hids: list = [16],
                 acts: list = ['relu'],
                 dropout: float = 0.5,
                 bn: bool = False,
                 normalize: bool = True,
                 bias: bool = True):
        r"""
        Parameters
        ----------
        in_feats : int, 
            the input dimmensions of model
        out_feats : int, 
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
        conv.append(GNNGUARDLayer())
        for hid, act in zip(hids, acts):
            conv.append(GCNConv(in_feats,
                                hid,
                                bias=bias, 
                                normalize=normalize))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            conv.append(GNNGUARDLayer())
            in_feats = hid
        conv.append(GCNConv(in_feats, out_feats, bias=bias, normalize=normalize))
        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.conv:
            if isinstance(layer, GNNGUARDLayer):
                edge_index, edge_weight = layer(x, edge_index)
            elif isinstance(layer, GCNConv):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x
