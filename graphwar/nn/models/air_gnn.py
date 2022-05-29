import torch.nn as nn
from graphwar.nn.layers import activations, AdaptiveConv, Sequential
from graphwar.utils import wrapper


class AirGNN(nn.Module):
    """Graph Neural Networks with Adaptive residual.

    Example:
    --------
    # AirGNN with one hidden layer
    >>> model = AirGNN(100, 10)
    # AirGNN with two hidden layers
    >>> model = AirGNN(100, 10, hids=[32, 16], acts=['relu', 'elu'])
    # AirGNN with two hidden layers, without activation at the first layer
    >>> model = AirGNN(100, 10, hids=[32, 16], acts=[None, 'relu'])
    """

    @wrapper
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hids: list = [64],
                 acts: list = ['relu'],
                 dropout: float = 0.8,
                 K: int = 3,
                 lambda_amp: float = 0.5,
                 bn: bool = False,
                 bias: bool = True):
        r"""
        Parameters
        ----------
        in_channels : int, 
            the input dimensions of model
        out_channels : int, 
            the output dimensions of model
        hids : list, optional
            the number of hidden units of each hidden layer, by default [64]
        acts : list, optional
            the activation function of each hidden layer, by default ['relu']
        dropout : float, optional
            the dropout ratio of model, by default 0.8
        bias : bool, optional
            whether to use bias in the layers, by default True
        bn: bool, optional
            whether to use `BatchNorm1d` after the convolution layer, by default False
        """

        super().__init__()
        assert len(hids) > 0

        lin = []
        for hid, act in zip(hids, acts):
            lin.append(nn.Dropout(dropout))
            lin.append(nn.Linear(in_channels, hid, bias=bias))
            if bn:
                lin.append(nn.BatchNorm1d(hid))
            lin.append(activations.get(act))
            in_channels = hid

        lin.append(nn.Dropout(dropout))
        lin.append(nn.Linear(in_channels, out_channels, bias=bias))

        self.prop = AdaptiveConv(K=K, lambda_amp=lambda_amp)

        self.lin = Sequential(*lin)

    def reset_parameters(self):
        self.prop.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.lin(x)
        return self.prop(x, edge_index, edge_weight)
