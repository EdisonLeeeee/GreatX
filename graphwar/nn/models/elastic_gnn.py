import torch.nn as nn
from graphwar.nn.layers import activations, ElasticConv, Sequential
from graphwar.utils import wrapper


class ElasticGNN(nn.Module):
    r"""Graph Neural Networks with elastic 
    message passing (ElasticGNN) from the `"Elastic Graph Neural 
    Networks" <https://arxiv.org/abs/2107.06996>`_
    paper (ICML'21)

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
    lambda1 : float, optional
        trade-off hyperparameter, by default 3
    lambda2 : float, optional
        trade-off hyperparameter, by default 3
    L21 : bool, optional
        whether to use row-wise projection 
        on the l2 ball of radius λ1., by default True
    cached : bool, optional
        whether to cache the incident matrix, by default True     
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
    >>> # ElasticGNN with one hidden layer
    >>> model = ElasticGNN(100, 10)

    >>> # ElasticGNN with two hidden layers
    >>> model = ElasticGNN(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # ElasticGNN with two hidden layers, without activation at the first layer
    >>> model = ElasticGNN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # ElasticGNN with very deep architectures, each layer has elu as activation function
    >>> model = ElasticGNN(100, 10, hids=[16]*8, acts=['elu'])

    See also
    --------
    :class:`graphwar.nn.layers.ElasticGNN`    

    """

    @wrapper
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hids: list = [16],
                 acts: list = ['relu'],
                 K: int = 3,
                 lambda1: float = 3,
                 lambda2: float = 3,
                 cached: bool = True,
                 dropout: float = 0.8,
                 bias: bool = True,
                 bn: bool = False):

        super().__init__()

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

        self.prop = ElasticConv(K=K,
                                lambda1=lambda1,
                                lambda2=lambda2,
                                L21=True,
                                cached=cached)

        self.lin = Sequential(*lin)

    def reset_parameters(self):
        self.prop.reset_parameters()
        self.lin.reset_parameters()

    def cache_clear(self):
        """Clear cached inputs or intermediate results."""
        self.prop._cached_inc = None
        return self

    def forward(self, x, edge_index, edge_weight=None):
        x = self.lin(x)
        return self.prop(x, edge_index, edge_weight)
