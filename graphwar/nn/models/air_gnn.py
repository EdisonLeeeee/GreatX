import torch.nn as nn
from graphwar.nn.layers import activations, AdaptiveConv, Sequential
from graphwar.utils import wrapper


class AirGNN(nn.Module):
    r"""Graph Neural Networks with Adaptive residual (AirGNN) 
    from the `"Graph Neural Networks with Adaptive Residual" 
    <https://openreview.net/forum?id=hfkER_KJiNw>`_
    paper (NeurIPS'21)

    Parameters
    ----------
    in_channels : int, 
        the input dimensions of model
    out_channels : int, 
        the output dimensions of model
    hids : list, optional
        the number of hidden units for each hidden layer, by default [64]
    acts : list, optional
        the activation function for each hidden layer, by default ['relu']
    K : int, optional
        the number of propagation steps during message passing, by default 3
    lambda_amp : float, optional
        trade-off for adaptive message passing, by default 0.1        
    dropout : float, optional
        the dropout ratio of model, by default 0.8
    bias : bool, optional
        whether to use bias in the layers, by default True
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer, by default False         

    Note
    ----
    It is convenient to extend the number of layers with different or the same
    hidden units (activation functions) using :meth:`graphwar.utils.wrapper`. 

    See Examples below:

    Examples
    --------
    >>> # AirGNN with one hidden layer
    >>> model = AirGNN(100, 10)

    >>> # AirGNN with two hidden layers
    >>> model = AirGNN(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # AirGNN with two hidden layers, without activation at the first layer
    >>> model = AirGNN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # AirGNN with very deep architectures, each layer has elu as activation function
    >>> model = AirGNN(100, 10, hids=[16]*8, acts=['elu'])

    See also
    --------
    :class:`graphwar.nn.layers.AdaptiveConv`    

    """

    @wrapper
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hids: list = [64],
                 acts: list = ['relu'],
                 K: int = 3,
                 lambda_amp: float = 0.5,
                 dropout: float = 0.8,
                 bias: bool = True,
                 bn: bool = False):

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
