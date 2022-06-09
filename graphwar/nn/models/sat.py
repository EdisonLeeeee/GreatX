import torch
import torch.nn as nn

from graphwar.utils import wrapper
from graphwar.nn.layers import SATConv, Sequential, activations


class SAT(nn.Module):
    r"""Graph Convolution Network with 
    Spectral Adversarial Training (SAT) from the `"Spectral Adversarial 
    Training for Robust Graph Neural Network"
    <https://arxiv.org>`_ paper (arXiv'22)

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
        whether to use bias in the layers, by default False
    normalize : bool, optional
        whether to compute symmetric normalization
        coefficients on the fly, by default True              
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer, by default False         

    Note
    ----
    It is convenient to extend the number of layers with different or the same
    hidden units (activation functions) using :meth:`graphwar.utils.wrapper`. 

    See Examples below:

    Examples
    --------
    >>> # SAT with one hidden layer
    >>> model = SAT(100, 10)

    >>> # SAT with two hidden layers
    >>> model = SAT(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # SAT with two hidden layers, without activation at the first layer
    >>> model = SAT(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # SAT with very deep architectures, each layer has elu as activation function
    >>> model = SAT(100, 10, hids=[16]*8, acts=['elu'])

    See also
    --------
    :class:`graphwar.nn.layers.SATConv`    

    """
    @wrapper
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hids: list = [16],
                 acts: list = ['relu'],
                 dropout: float = 0.5,
                 bias: bool = False,
                 normalize: bool = True,
                 bn: bool = False):
        super().__init__()

        conv = []
        assert len(hids) == len(acts)
        for hid, act in zip(hids, acts):
            conv.append(SATConv(in_channels,
                                hid,
                                bias=bias,
                                normalize=normalize))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_channels = hid
        conv.append(SATConv(in_channels, out_channels,
                    bias=bias, normalize=normalize))
        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)


# class SAT(nn.Module):
#     @wrapper
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  hids: list = [],
#                  acts: list = [],
#                  dropout: float = 0.5,
#                  K: int = 5,
#                  alpha: float = 0.1,
#                  normalize: bool = True,
#                  bn: bool = False,
#                  bias: bool = False):
#         super().__init__()

#         conv = []
#         for i, (hid, act) in enumerate(zip(hids, acts)):
#             if i == 0:
#                 conv.append(SATConv(in_channels,
#                                     hid,
#                                     bias=bias,
#                                     K=K,
#                                     normalize=normalize,
#                                     alpha=alpha))
#             else:
#                 conv.append(nn.Linear(in_channels, hid, bias=bias))
#             if bn:
#                 conv.append(nn.BatchNorm1d(hid))
#             conv.append(activations.get(act))
#             conv.append(nn.Dropout(dropout))
#             in_channels = hid

#         if not hids:
#             conv.append(SATConv(in_channels,
#                                 out_channels,
#                                 bias=bias,
#                                 K=K,
#                                 normalize=normalize,
#                                 alpha=alpha))
#         else:
#             conv.append(nn.Linear(in_channels, out_channels, bias=bias))

#         self.conv = Sequential(*conv)

#     def reset_parameters(self):
#         self.conv.reset_parameters()

#     def forward(self, x, edge_index, edge_weight=None):
#         return self.conv(x, edge_index, edge_weight)
