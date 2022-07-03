import torch
import math
import torch.nn as nn

from greatx.nn.layers import GCNConv, Sequential, activations
from greatx.utils import wrapper


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))

    def uniform(self, size, tensor):
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
