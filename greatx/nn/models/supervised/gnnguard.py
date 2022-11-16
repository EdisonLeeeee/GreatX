import torch.nn as nn

from greatx.defense import GNNGUARD as GNNGUARDLayer
from greatx.nn.layers import GCNConv, Sequential, activations
from greatx.utils import wrapper


class GNNGUARD(nn.Module):
    r"""Graph Convolution Network (GCN) with
    :class:`~greatx.defense.GNNGUARD` from the `"GNNGUARD:
    Defending Graph Neural Networks against Adversarial Attacks"
    <https://arxiv.org/abs/2006.08149>`_ paper (NeurIPS'20)

    Parameters
    ----------
    in_channels : int,
        the input dimensions of model
    out_channels : int,
        the output dimensions of model
    hids : list, optional
        the number of hidden units for each hidden layer,
        by default [16]
    acts : list, optional
        the activation function for each hidden layer,
        by default ['relu']
    dropout : float, optional
        the dropout ratio of model, by default 0.5
    bias : bool, optional
        whether to use bias in the layers, by default True
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer,
        by default False

    Examples
    --------
    >>> # GNNGUARD with one hidden layer
    >>> model = GNNGUARD(100, 10)

    >>> # GNNGUARD with two hidden layers
    >>> model = GNNGUARD(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # GNNGUARD with two hidden layers, without first activation
    >>> model = GNNGUARD(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # GNNGUARD with deep architectures, each layer has elu activation
    >>> model = GNNGUARD(100, 10, hids=[16]*8, acts=['elu'])

    See also
    --------
    :class:`~greatx.defense.GNNGUARD`
    :class:`~greatx.nn.models.supervised.GCN`
    """
    @wrapper
    def __init__(self, in_channels: int, out_channels: int, hids: list = [16],
                 acts: list = ['relu'], dropout: float = 0.5, bn: bool = False,
                 normalize: bool = True, bias: bool = True):

        super().__init__()

        conv = []
        # Add self-loops in the first input layer
        conv.append(GNNGUARDLayer(add_self_loops=True))
        for hid, act in zip(hids, acts):
            conv.append(
                GCNConv(in_channels, hid, bias=bias, add_self_loops=False,
                        normalize=normalize))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(GNNGUARDLayer())
            conv.append(nn.Dropout(dropout))
            in_channels = hid
        conv.append(
            GCNConv(in_channels, out_channels, add_self_loops=False, bias=bias,
                    normalize=normalize))
        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        for layer in self.conv:
            if isinstance(layer, GNNGUARDLayer):
                edge_index, edge_weight = layer(x, edge_index)
            elif isinstance(layer, GCNConv):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x
