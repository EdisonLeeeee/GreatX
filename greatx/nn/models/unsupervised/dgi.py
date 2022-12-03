import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from greatx.nn.layers import GCNConv, Sequential, activations
from greatx.utils import wrapper

bce = F.binary_cross_entropy_with_logits


class Discriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.weight = nn.Parameter(Tensor(in_channels, in_channels))

    @staticmethod
    def uniform(size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, x: Tensor, summary: Tensor) -> Tensor:
        """"""
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
    hids : List[int], optional
        the number of hidden units for each hidden layer, by default [512]
    acts : List[str], optional
        the activation function for each hidden layer, by default ['prelu']
    dropout : float, optional
        the dropout ratio of model, by default 0.0
    bias : bool, optional
        whether to use bias in the layers, by default True
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer,
        by default False

    Examples
    --------
    >>> # DGI with one hidden layer
    >>> model = DGI(100)

    >>> # DGI with two hidden layers
    >>> model = DGI(100, hids=[32, 16], acts=['relu', 'elu'])

    >>> # DGI with two hidden layers, without first activation
    >>> model = DGI(100, hids=[32, 16], acts=[None, 'relu'])

    >>> # DGI with deep architectures, each layer has elu activation
    >>> model = DGI(100, hids=[16]*8, acts=['elu'])

    Reference:

    * Author's code: https://github.com/PetarV-/DGI

    """
    @wrapper
    def __init__(
        self,
        in_channels: int,
        hids: List[int] = [512],
        acts: List[str] = ['prelu'],
        dropout: float = 0.,
        bias: bool = True,
        bn: bool = False,
    ):

        super().__init__()

        encoder = []
        for hid, act in zip(hids, acts):
            encoder.append(GCNConv(in_channels, hid, bias=bias))
            if bn:
                encoder.append(nn.BatchNorm1d(hid))
            encoder.append(activations.get(act))
            encoder.append(nn.Dropout(dropout))
            in_channels = hid

        self.encoder = Sequential(*encoder)
        self.discriminator = Discriminator(in_channels)
        self.reset_parameters()

    @staticmethod
    def corruption(x: Tensor) -> Tensor:
        return x[torch.randperm(x.size(0))]

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.discriminator.reset_parameters()

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        z = self.encoder(x, edge_index, edge_weight)
        return z

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """"""
        z1 = self.encode(x, edge_index, edge_weight)  # view1
        z2 = self.encode(self.corruption(x), edge_index, edge_weight)  # view2
        summary = torch.sigmoid(z1.mean(dim=0))  # global

        # global-local contrasting
        pos = self.discriminator(z1, summary).squeeze()
        # global-local contrasting
        neg = self.discriminator(z2, summary).squeeze()

        return pos, neg

    def loss(self, postive: Tensor, negative: Tensor) -> Tensor:
        loss = bce(postive, postive.new_ones(postive.size(0))) + \
            bce(negative, negative.new_zeros(negative.size(0)))
        return loss
