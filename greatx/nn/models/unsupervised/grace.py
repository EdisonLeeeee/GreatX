from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv

from greatx.nn.layers import GCNConv, Sequential, activations
from greatx.nn.models import MLP
from greatx.utils import wrapper

try:
    from torch_geometric.utils import dropout_edge, mask_feature
except ImportError:
    dropout_edge = mask_feature = None


class GRACE(torch.nn.Module):
    r"""GRAph Contrastive rEpresentation learning (GRACE) from the
    `"Deep Graph Contrastive Representation Learning"
    <https://arxiv.org/abs/2006.04131>`_ paper (ICML'20)

    Parameters
    ----------
    in_channels : int,
        the input dimensions of model
    hids : List[int], optional
        the number of hidden units for each hidden layer, by default [128]
    acts : List[str], optional
        the activation function for each hidden layer, by default ['prelu']
    project_hids : List[int], optional
        the projection dimensions of model, by default [128]
    tau : float, optional
        the temperature coefficient of softmax, by default 0.5
    dropout : float, optional
        the dropout ratio of model, by default 0.0
    bias : bool, optional
        whether to use bias in the layers, by default True
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer,
        by default False
    drop_edge_1 : float, optional
        the dropout ratio of edges for the first view, by default 0.8
    drop_edge_1 : float, optional
        the dropout ratio of edges for the second view, by default 0.7
    drop_feat_1 : float, optional
        the dropout ratio of features for the first view, by default 0.4
    drop_feat_2 : float, optional
        the dropout ratio of features for the second view, by default 0.3


    Examples
    --------
    >>> # GRACE with one hidden layer
    >>> model = GRACE(100)

    >>> # GRACE with two hidden layers
    >>> model = GRACE(100, hids=[32, 16], acts=['relu', 'elu'])

    >>> # GRACE with two hidden layers, without first activation
    >>> model = GRACE(100, hids=[32, 16], acts=[None, 'relu'])

    >>> # GRACE with deep architectures, each layer has elu activation
    >>> model = GRACE(100, hids=[16]*8, acts=['elu'])

    Reference:

    * Author's code: https://github.com/CRIPAC-DIG/GRACE

    """
    @wrapper
    def __init__(
        self,
        in_channels: int,
        hids: List[int] = [128],
        acts: List[str] = ['prelu'],
        project_hids: List[int] = [128],
        dropout: float = 0.,
        tau: float = 0.5,
        bias: bool = True,
        bn: bool = False,
        drop_edge_1: float = 0.8,
        drop_edge_2: float = 0.7,
        drop_feat_1: float = 0.4,
        drop_feat_2: float = 0.3,
    ):

        super().__init__()

        if dropout_edge is None:
            # TODO: support them
            raise ImportError(
                "Please install the latest version of `torch_geometric`.")

        encoder = []
        for hid, act in zip(hids, acts):
            encoder.append(GCNConv(
                in_channels,
                hid,
                bias=bias,
            ))
            if bn:
                encoder.append(nn.BatchNorm1d(hid))
            encoder.append(activations.get(act))
            encoder.append(nn.Dropout(dropout))
            in_channels = hid

        self.encoder = Sequential(*encoder)
        self.decoder = MLP(in_channels, in_channels, project_hids, dropout=0.)
        self.tau = tau
        self.drop_edge_1 = drop_edge_1
        self.drop_edge_2 = drop_edge_2
        self.drop_feat_1 = drop_feat_1
        self.drop_feat_2 = drop_feat_2

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

        edge_index_1, mask_1 = dropout_edge(edge_index, p=self.drop_edge_1)
        edge_index_2, mask_2 = dropout_edge(edge_index, p=self.drop_edge_2)
        if edge_weight is not None:
            edge_weight_1 = edge_index[:, mask_1]
            edge_weight_2 = edge_index[:, mask_2]
        else:
            edge_weight_1 = edge_weight_2 = None

        x_1 = mask_feature(x, self.drop_feat_1)[0]
        x_2 = mask_feature(x, self.drop_feat_2)[0]

        z1 = self.encoder(x_1, edge_index_1, edge_weight_1)
        z2 = self.encoder(x_2, edge_index_2, edge_weight_2)
        h1 = self.decoder(z1)
        h2 = self.decoder(z2)

        return h1, h2

    def sim(self, z1: Tensor, z2: Tensor) -> Tensor:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        refl_sim = torch.exp(self.sim(z1, z1) / self.tau)
        between_sim = torch.exp(self.sim(z1, z2) / self.tau)
        a = between_sim.diag()
        b = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        c = -torch.log(a / b)
        return c.mean()
