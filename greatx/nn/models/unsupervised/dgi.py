import torch
import math
import torch.nn as nn

from greatx.nn.layers import GCNConv, Sequential, activations
from greatx.utils import wrapper


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, in_channels))

    @staticmethod
    def uniform(size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, x, summary):
        x = torch.matmul(x, torch.matmul(self.weight, summary))
        return x


class DGI(nn.Module):
    r"""Deep Graph Infomax (DGI) from the 
    `"Deep Graph Infomax"
    <https://arxiv.org/abs/1809.10341>`_ paper (ICLR'19)

    Parameters
    ----------
    in_channels : int, 
        the input dimensions of model
    hids : list, optional
        the number of hidden units for each hidden layer, by default [512]
    acts : list, optional
        the activation function for each hidden layer, by default ['prelu']
    dropout : float, optional
        the dropout ratio of model, by default 0.0
    bias : bool, optional
        whether to use bias in the layers, by default True
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer, by default False         
    normalize : bool, optional
        whether to compute symmetric normalization
        coefficients on the fly, by default True        

    Note
    ----
    It is convenient to extend the number of layers with different or the same
    hidden units (activation functions) using :func:`greatx.utils.wrapper`. 

    See Examples below:

    Examples
    --------
    >>> # DGI with one hidden layer
    >>> model = DGI(100)

    >>> # DGI with two hidden layers
    >>> model = DGI(100, hids=[32, 16], acts=['relu', 'elu'])

    >>> # DGI with two hidden layers, without activation at the first layer
    >>> model = DGI(100, hids=[32, 16], acts=[None, 'relu'])

    >>> # DGI with very deep architectures, each layer has elu as activation function
    >>> model = DGI(100, hids=[16]*8, acts=['elu'])

    Reference:

    * Author's code: https://github.com/PetarV-/DGI

    """
    @wrapper
    def __init__(self,
                 in_channels: int,
                 hids: list = [512],
                 acts: list = ['prelu'],
                 dropout: float = 0.,
                 bias: bool = True,
                 bn: bool = False,
                 normalize: bool = True):

        super().__init__()

        encoder = []
        for hid, act in zip(hids, acts):
            encoder.append(GCNConv(in_channels,
                                   hid,
                                   bias=bias,
                                   normalize=normalize))
            if bn:
                encoder.append(nn.BatchNorm1d(hid))
            encoder.append(activations.get(act))
            encoder.append(nn.Dropout(dropout))
            in_channels = hid

        self.encoder = Sequential(*encoder)
        self.discriminator = Discriminator(in_channels)
        self.reset_parameters()

    @staticmethod
    def corruption(x):
        return x[torch.randperm(x.size(0))]

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.discriminator.reset_parameters()

    def encode(self, x, edge_index, edge_weight=None):
        z = self.encoder(x, edge_index, edge_weight)
        return z

    def forward(self, x, edge_index, edge_weight=None):
        z1 = self.encode(x, edge_index, edge_weight)  # view1
        z2 = self.encode(self.corruption(x), edge_index, edge_weight)  # view2
        summary = torch.sigmoid(z1.mean(dim=0))  # global

        pos = self.discriminator(z1, summary).squeeze()  # global-local contrasting
        neg = self.discriminator(z2, summary).squeeze()  # global-local contrasting

        return pos, neg
