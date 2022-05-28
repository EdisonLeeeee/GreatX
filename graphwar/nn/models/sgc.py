import torch.nn as nn

from graphwar.utils import wrapper
from graphwar.nn.layers import SGConv, Sequential, activations

class SGC(nn.Module):
    """Simplifying Graph Convolution layer from paper `Simplifying Graph
    Convolutional Networks <https://arxiv.org/pdf/1902.07153.pdf>`__.

    Example
    -------
    # SGC model without hidden layers (by default)
    >>> model = SGC(100, 10)
    # SGC with one hidden layers
    >>> model = SGC(100, 10, hids=[16], acts=['relu'])    
    """
    
    @wrapper
    def __init__(self,
                 in_feats,
                 out_feats,
                 hids: list = [],
                 acts: list = [],
                 dropout: float = 0.,
                 K: int = 2,
                 bn: bool = False,
                 bias: bool = True,
                 cached: bool = True):
        super().__init__()

        conv = []
        for i, (hid, act) in enumerate(zip(hids, acts)):
            if i == 0:
                conv.append(SGConv(in_feats,
                                   hid,
                                   bias=bias,
                                   K=K,
                                   cached=cached))
            else:
                conv.append(nn.Linear(in_feats, hid, bias=bias))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))                
            conv.append(nn.Dropout(dropout))
            in_feats = hid
            
        if not hids:
            conv.append(SGConv(in_feats,
                      out_feats,
                      bias=bias,
                      K=K,
                      cached=cached))
        else:
            conv.append(nn.Linear(in_feats, out_feats, bias=bias))
            
        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def cache_clear(self):
        for layer in self.conv:
            if hasattr(layer, 'cache_clear'):
                layer.cache_clear()
        return self
    
    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)