from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv

from greatx.nn.layers import GCNConv, Sequential, activations
from greatx.utils import wrapper

try:
    from torch_geometric.utils import dropout_edge, mask_feature
except ImportError:
    dropout_edge = mask_feature = None


class CCA_SSG(torch.nn.Module):
    r"""CCA-SSG model from the
    `"From Canonical Correlation Analysis to
    Self-supervised Graph Neural Networks"
    <https://arxiv.org/abs/2106.12484>`_ paper (NeurIPS'21)

    Parameters
    ----------
    in_channels : int,
        the input dimensions of model
    hids : List[int], optional
        the number of hidden units for each hidden layer,
        by default [512, 512]
    acts : List[str], optional
        the activation function for each hidden layer,
        by default ['prelu', 'prelu']
    project_hids : List[int], optional
        the projection dimensions of model, by default [512, 512]
    lambd : float, optional
        the trade-off of the loss, by default 1e-3
    dropout : float, optional
        the dropout ratio of model, by default 0.0
    bias : bool, optional
        whether to use bias in the layers, by default True
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer,
        by default False
    drop_edge : float, optional
        the dropout ratio of edges for contrasting, by default 0.2
    drop_feat : float, optional
        the dropout ratio of features for contrasting, by default 0.2


    Examples
    --------
    >>> # CCA_SSG with one hidden layer
    >>> model = CCA_SSG(100)

    >>> # CCA_SSG with two hidden layers
    >>> model = CCA_SSG(100, hids=[32, 16], acts=['relu', 'elu'])

    >>> # CCA_SSG with two hidden layers, without first activation
    >>> model = CCA_SSG(100, hids=[32, 16], acts=[None, 'relu'])

    >>> # CCA_SSG with deep architectures, each layer has elu activation
    >>> model = CCA_SSG(100, hids=[16]*8, acts=['elu'])

    Reference:

    * Author's code: https://github.com/hengruizhang98/CCA-SSG

    """
    @wrapper
    def __init__(
        self,
        in_channels: int,
        hids: List[int] = [512, 512],
        acts: List[str] = ['prelu', 'prelu'],
        dropout: float = 0.,
        lambd: float = 1e-3,
        bias: bool = True,
        bn: bool = False,
        drop_edge: float = 0.2,
        drop_feat: float = 0.2,
    ):

        super().__init__()

        if dropout_edge is None:
            # TODO: support them
            raise ImportError(
                "Please install the latest version of `torch_geometric`.")

        encoder = []
        for hid, act in zip(hids, acts):
            encoder.append(GCNConv(in_channels, hid, bias=bias))
            if bn:
                encoder.append(nn.BatchNorm1d(hid))
            encoder.append(activations.get(act))
            encoder.append(nn.Dropout(dropout))
            in_channels = hid

        self.encoder = Sequential(*encoder)
        self.drop_edge = drop_edge
        self.drop_feat = drop_feat
        self.lambd = lambd

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

        edge_index1, mask1 = dropout_edge(edge_index, p=self.drop_edge)
        edge_index2, mask2 = dropout_edge(edge_index, p=self.drop_edge)

        if edge_weight is not None:
            edge_weight1 = edge_weight[mask1]
            edge_weight2 = edge_weight[mask2]
        else:
            edge_weight1 = edge_weight2 = None

        x1 = mask_feature(x, self.drop_feat)[0]
        x2 = mask_feature(x, self.drop_feat)[0]

        h1 = self.encoder(x1, edge_index1, edge_weight1)
        h2 = self.encoder(x2, edge_index2, edge_weight2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        return z1, z2

    def sim(self, z1: Tensor, z2: Tensor) -> Tensor:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        c = z1.t() @ z2
        c1 = z1.t() @ z1
        c2 = z2.t() @ z2

        N = z1.size(0)
        c = c / N
        c1 = c1 / N
        c2 = c2 / N

        loss_inv = -torch.diagonal(c).sum()
        iden = torch.eye(c.size(0), device=c.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        loss = loss_inv + self.lambd * (loss_dec1 + loss_dec2)
        return loss
