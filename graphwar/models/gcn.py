import torch
import torch.nn as nn
from graphwar.nn import Sequential, activations, GCNConv
from graphwar.config import Config
from graphwar.utils import wrapper

_EDGE_WEIGHT = Config.edge_weight


class GCN(nn.Module):
    """Graph Convolution Network (GCN)

    Example
    -------
    # GCN with one hidden layer
    >>> model = GCN(100, 10)
    # GCN with two hidden layers
    >>> model = GCN(100, 10, hids=[32, 16], acts=['relu', 'elu'])
    # GCN with two hidden layers, without activation at the first layer
    >>> model = GCN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    Note
    ----
    please make sure `hids` and `acts` are both `list` or `tuple` and
    `len(hids)==len(acts)`.

    """
    
    @wrapper
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hids: list = [16],
                 acts: list = ['relu'],
                 dropout: float = 0.5,
                 bn: bool = False,
                 bias: bool = True,
                 norm: str = 'both'):
        r"""
        Parameters
        ----------
        in_features : int, 
            the input dimmensions of model
        out_features : int, 
            the output dimensions of model
        hids : list, optional
            the number of hidden units of each hidden layer, by default [16]
        acts : list, optional
            the activaction function of each hidden layer, by default ['relu']
        dropout : float, optional
            the dropout ratio of model, by default 0.5
        bias : bool, optional
            whether to use bias in the layers, by default True
        bn: bool, optional
            whether to use `BatchNorm1d` after the convolution layer, by default False
        norm : str, optional
            How to apply the normalizer.  Can be one of the following values:

            * ``both``, where the messages are scaled with :math:`1/c_{ji}`, 
            where :math:`c_{ji}` is the product of the square root of node degrees
            (i.e.,  :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`).

            * ``square``, where the messages are scaled with :math:`1/c_{ji}^2`, where
            :math:`c_{ji}` is defined as above.

            * ``right``, to divide the aggregated messages by each node's in-degrees,
            which is equivalent to averaging the received messages.

            * ``none``, where no normalization is applied.

            * ``left``, to divide the messages sent out from each node by its out-degrees,
            equivalent to random walk normalization.            
        """

        super().__init__()

        conv = []
        assert len(hids) == len(acts)
        for hid, act in zip(hids, acts):
            conv.append(GCNConv(in_features,
                                hid,
                                bias=bias, norm=norm,
                                activation=None))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(GCNConv(in_features, out_features, bias=bias, norm=norm))
        self.conv = Sequential(*conv, loc=1)  # `loc=1` specifies the location of features.

    def reset_parameters(self):
        for conv in self.conv:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()

    def forward(self, g, feat, edge_weight=None):

        if torch.is_tensor(g):
            return self.forward_by_adjacency_matrix(g, feat)

        if edge_weight is None:
            edge_weight = g.edata.get(_EDGE_WEIGHT, edge_weight)

        return self.conv(g, feat, edge_weight=edge_weight)

    def forward_by_adjacency_matrix(self, adj_matrix, feat):
        assert feat is not None
        for conv in self.conv:
            if isinstance(conv, GCNConv):
                if conv.weight is not None:
                    feat = feat @ conv.weight
                feat = adj_matrix @ feat
                if conv.bias is not None:
                    feat = feat + conv.bias
                if conv._activation is not None:
                    feat = conv._activation(feat)
            else:
                feat = conv(feat)
        return feat
