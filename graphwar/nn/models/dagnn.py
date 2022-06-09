import torch.nn as nn
from torch.nn import Linear

from graphwar.nn.layers import Sequential, activations, DAGNNConv
from graphwar.utils import wrapper


class DAGNN(nn.Module):
    r"""The DAGNN operator from the `"Towards Deeper Graph Neural 
    Networks" <https://arxiv.org/abs/2007.09296>`_
    paper (KDD'20)

    Parameters
    ----------
    in_channels : int, 
        the input dimensions of model
    out_channels : int, 
        the output dimensions of model
    hids : list, optional
        the number of hidden units for each hidden layer, by default [64]
    K : int, optional
        the number of propagation steps, by default 10    
    acts : list, optional
        the activation function for each hidden layer, by default ['relu']
    dropout : float, optional
        the dropout ratio of model, by default 0.5
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
    >>> # DAGNN with one hidden layer
    >>> model = DAGNN(100, 10)

    >>> # DAGNN with two hidden layers
    >>> model = DAGNN(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # DAGNN with two hidden layers, without activation at the first layer
    >>> model = DAGNN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # DAGNN with very deep architectures, each layer has elu as activation function
    >>> model = DAGNN(100, 10, hids=[16]*8, acts=['elu'])

    See also
    --------
    :class:`graphwar.nn.layers.DAGNNConv`    

    """

    @wrapper
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hids: list = [64],
                 acts: list = ['relu'],
                 dropout: float = 0.5,
                 K: int = 10,
                 bn: bool = False,
                 bias: bool = True):

        super().__init__()
        assert len(hids) > 0

        lin = []
        for hid, act in zip(hids, acts):
            lin.append(nn.Dropout(dropout))
            lin.append(Linear(in_channels, hid, bias=bias))
            if bn:
                lin.append(nn.BatchNorm1d(hid))
            lin.append(activations.get(act))
            in_channels = hid

        lin.append(nn.Dropout(dropout))
        lin.append(Linear(in_channels, out_channels, bias=bias))

        self.prop = DAGNNConv(out_channels, 1, K=K)

        self.lin = Sequential(*lin)

    def reset_parameters(self):
        self.prop.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.lin(x)
        return self.prop(x, edge_index, edge_weight)
