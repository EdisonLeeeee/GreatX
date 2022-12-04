from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import Linear

from greatx.functional import spmm
from greatx.nn.layers import GCNConv, Sequential, activations
from greatx.nn.layers.gcn_conv import make_gcn_norm
from greatx.utils import wrapper

try:
    from torch_geometric.utils import mask_feature
except ImportError:
    mask_feature = None

bce = F.binary_cross_entropy_with_logits


class GGD(nn.Module):
    r"""Graph Group Discrimination (GGD) from the
    `"Rethinking and Scaling Up Graph Contrastive Learning:
    An Extremely Efficient Approach with Group Discrimination"
    <https://arxiv.org/abs/2206.01535>`_ paper (NeurIPS'22)

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
    drop_feat : float, optional
        the dropout ratio of features for contrasting, by default 0.2

    Examples
    --------
    >>> # GGD with one hidden layer
    >>> model = GGD(100)

    >>> # GGD with two hidden layers
    >>> model = GGD(100, hids=[32, 16], acts=['relu', 'elu'])

    >>> # GGD with two hidden layers, without first activation
    >>> model = GGD(100, hids=[32, 16], acts=[None, 'relu'])

    >>> # GGD with deep architectures, each layer has elu activation
    >>> model = GGD(100, hids=[16]*8, acts=['elu'])

    Reference:

    * Author's code: https://github.com/zyzisastudyreallyhardguy/Graph-Group-Discrimination # noqa

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
        drop_feat: float = 0.2,
    ):

        super().__init__()

        if mask_feature is None:
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
        self.discriminator = Linear(in_channels, in_channels, bias=bias)
        self.drop_feat = drop_feat
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
        k: int = 0,
    ) -> Tensor:
        z = self.encoder(x, edge_index, edge_weight)

        if not self.training:
            edge_index, edge_weight = make_gcn_norm(edge_index, edge_weight,
                                                    num_nodes=x.size(0),
                                                    dtype=x.dtype,
                                                    add_self_loops=True)
            h = z
            for _ in range(k):
                h = spmm(h, edge_index, edge_weight)
            if k:
                z += h
        return z

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """"""
        x = mask_feature(x, self.drop_feat)[0]
        z1 = self.encode(x, edge_index, edge_weight)  # view1
        z2 = self.encode(self.corruption(x), edge_index, edge_weight)  # view2

        pos = self.discriminator(z1).sum(1)
        neg = self.discriminator(z2).sum(1)

        return pos, neg

    def loss(self, postive: Tensor, negative: Tensor) -> Tensor:
        loss = bce(postive, postive.new_ones(postive.size(0))) + \
            bce(negative, negative.new_zeros(negative.size(0)))
        return loss
