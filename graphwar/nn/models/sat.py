import torch
import torch.nn as nn

from graphwar.utils import wrapper
from graphwar.nn.layers import SATConv, Sequential, activations


class SAT(nn.Module):

    @wrapper
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hids: list = [16],
                 acts: list = ['relu'],
                 dropout: float = 0.5,
                 bn: bool = False,
                 normalize: bool = True,
                 bias: bool = False):
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
