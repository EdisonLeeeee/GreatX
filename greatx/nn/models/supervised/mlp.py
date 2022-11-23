import torch
from torch import nn
from torch_geometric.nn.inits import zeros

from greatx.nn.layers import Sequential, activations
from greatx.utils import wrapper


class MLP(nn.Module):
    r"""Implementation of Multi-layer Perceptron (MLP) or
    Feed-forward Neural Network (FNN).

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
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the Linear layer,
        by default False

    Examples
    --------
    >>> # MLP with one hidden layer
    >>> model = MLP(100, 10)

    >>> # MLP with two hidden layers
    >>> model = MLP(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # MLP with two hidden layers, without first activation
    >>> model = MLP(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # MLP with deep architectures, each layer has elu activation
    >>> model = MLP(100, 10, hids=[16]*8, acts=['elu'])

    See also
    --------
    :class:`greatx.nn.models.supervised.LogisticRegression`

    """
    @wrapper
    def __init__(self, in_channels: int, out_channels: int, hids: list = [16],
                 acts: list = ['relu'], dropout: float = 0.5,
                 bias: bool = True, bn: bool = False):

        super().__init__()

        lin = []
        for hid, act in zip(hids, acts):
            lin.append(nn.Linear(in_channels, hid, bias=bias))
            if bn:
                lin.append(nn.BatchNorm1d(hid))
            lin.append(activations.get(act))
            lin.append(nn.Dropout(dropout))
            in_channels = hid
        lin.append(nn.Linear(in_channels, out_channels, bias=bias))
        self.lin = Sequential(*lin)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, *args, **kwargs):
        """"""
        return self.lin(x)


class LogisticRegression(nn.Module):
    r"""Simple logistic regression model for
    self-supervised/unsupervised learning.

    Parameters
    ----------
    in_channels : int,
        the input dimensions of model
    out_channels : int,
        the output dimensions of model
    bias : bool, optional
        whether to use bias in the layers, by default True

    See Examples below.

    Examples
    --------
    >>> # LogisticRegression without hidden layer
    >>> model = LogisticRegression(100, 10)

    See also
    --------
    :class:`greatx.nn.models.supervised.MLP`
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight.data)
        zeros(self.lin.bias)

    def forward(self, x, *args, **kwargs):
        """"""
        return self.lin(x)
