import torch
import torch.nn as nn

from graphwar.nn.layers import TAGConv, Sequential, activations
from graphwar.utils import wrapper


class TAGCN(nn.Module):
    """Topological Adaptive Graph Convolutional Networks 
    (`TAGCN <https://arxiv.org/abs/1710.10370>`__)

    Example
    -------
    # TAGCN with one hidden layer
    >>> model = TAGCN(100, 10)
    # TAGCN with two hidden layers
    >>> model = TAGCN(100, 10, hids=[32, 16], acts=['relu', 'elu'])
    # TAGCN with two hidden layers, without activation at the first layer
    >>> model = TAGCN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    """

    @wrapper
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hids: list = [16],
                 acts: list = ['relu'],
                 K: int = 2,
                 dropout: float = 0.5,
                 bn: bool = False,
                 normalize: bool = True,
                 bias: bool = True):
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
