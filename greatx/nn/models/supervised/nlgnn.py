from typing import List

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from greatx.nn.layers import GCNConv, Sequential, activations
from greatx.utils import wrapper


class NLGCN(nn.Module):
    r"""Non-Local Graph Neural Networks (NLGNN) with
    :class:`GCN` as backbone from the
    `"Non-Local Graph Neural Networks"
    <https://ieeexplore.ieee.org/document/9645300>`_ paper (TPAMI'22)

    Parameters
    ----------
    in_channels : int,
        the input dimensions of model
    out_channels : int,
        the output dimensions of model
    hids : List[int], optional
        the number of hidden units for each hidden layer, by default [16]
    acts : List[str], optional
        the activation function for each hidden layer, by default ['relu']
    kernel : int,
        the number of kernel used in :class:`nn.Conv1d`, by default 5
    dropout : float, optional
        the dropout ratio of model, by default 0.5
    bias : bool, optional
        whether to use bias in the layers, by default True
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer,
        by default False



    Examples
    --------
    >>> # NLGCN with one hidden layer
    >>> model = NLGCN(100, 10)

    >>> # NLGCN with two hidden layers
    >>> model = NLGCN(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # NLGCN with two hidden layers, without first activation
    >>> model = NLGCN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # NLGCN with deep architectures, each layer has elu activation
    >>> model = NLGCN(100, 10, hids=[16]*8, acts=['elu'])

    See also
    --------
    :class:`greatx.nn.models.supervised.NLMLP`
    :class:`greatx.nn.models.supervised.NLGAT`

    Reference:

    * https://github.com/divelab/Non-Local-GNN

    """
    @wrapper
    def __init__(self, in_channels: int, out_channels: int,
                 hids: List[int] = [16], acts: List[str] = ['relu'],
                 kernel: int = 5, dropout: float = 0.5, bn: bool = False,
                 normalize: bool = True, bias: bool = True):

        super().__init__()
        conv = []
        assert len(hids) == len(acts)
        for hid, act in zip(hids, acts):
            conv.append(
                GCNConv(in_channels, hid, bias=bias, normalize=normalize))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_channels = hid

        conv.append(
            GCNConv(in_channels, out_channels, bias=bias, normalize=normalize))
        self.conv = Sequential(*conv)

        self.proj = nn.Linear(out_channels, 1)
        self.conv1d_1 = nn.Conv1d(out_channels, out_channels, kernel,
                                  padding=int((kernel - 1) / 2))
        self.conv1d_2 = nn.Conv1d(out_channels, out_channels, kernel,
                                  padding=int((kernel - 1) / 2))
        self.lin = nn.Linear(2 * out_channels, out_channels)
        self.conv1d_dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.proj.reset_parameters()
        self.conv1d_1.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x1 = self.conv(x, edge_index, edge_weight)
        g_score = self.proj(x1)  # [num_nodes, 1]
        g_score_sorted, sort_idx = torch.sort(g_score, dim=0)
        _, inverse_idx = torch.sort(sort_idx, dim=0)

        sorted_x = g_score_sorted * x1[sort_idx].squeeze()
        sorted_x = torch.transpose(sorted_x, 0, 1).unsqueeze(
            0)  # [1, dataset.num_classes, num_nodes]
        sorted_x = self.conv1d_1(sorted_x).relu()
        sorted_x = self.conv1d_dropout(sorted_x)
        sorted_x = self.conv1d_2(sorted_x)
        # [num_nodes, dataset.num_classes]
        sorted_x = torch.transpose(sorted_x.squeeze(), 0, 1)
        # [num_nodes, dataset.num_classes]
        x2 = sorted_x[inverse_idx].squeeze()

        out = torch.cat([x1, x2], dim=1)
        out = self.lin(out)
        return out


class NLMLP(nn.Module):
    r"""Non-Local Graph Neural Networks (NLGNN) with
    :class:`MLP` as backbone from the
    `"Non-Local Graph Neural Networks"
    <https://ieeexplore.ieee.org/document/9645300>`_ paper (TPAMI'22)

    Parameters
    ----------
    in_channels : int,
        the input dimensions of model
    out_channels : int,
        the output dimensions of model
    hids : List[int], optional
        the number of hidden units for each hidden layer, by default [32]
    acts : List[str], optional
        the activation function for each hidden layer, by default ['relu']
    kernel : int,
        the number of kernel used in :class:`nn.Conv1d`, by default 5
    dropout : float, optional
        the dropout ratio of model, by default 0.5
    bias : bool, optional
        whether to use bias in the layers, by default True
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer,
        by default False
    normalize : bool, optional
        whether to compute symmetric normalization
        coefficients on the fly, by default True

    Examples
    --------
    >>> # NLGCN with one hidden layer
    >>> model = NLGCN(100, 10)

    >>> # NLGCN with two hidden layers
    >>> model = NLGCN(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # NLGCN with two hidden layers, without first activation
    >>> model = NLGCN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # NLGCN with deep architectures, each layer has elu activation
    >>> model = NLGCN(100, 10, hids=[16]*8, acts=['elu'])

    See also
    --------
    :class:`greatx.nn.models.supervised.NLGCN`
    :class:`greatx.nn.models.supervised.NLGAT`

    Reference:

    * https://github.com/divelab/Non-Local-GNN

    """
    @wrapper
    def __init__(self, in_channels: int, out_channels: int,
                 hids: List[int] = [32], acts: List[str] = ['relu'],
                 kernel: int = 5, dropout: float = 0.5, bias: bool = True,
                 bn: bool = False):

        super().__init__()
        conv = []
        assert len(hids) == len(acts)
        for hid, act in zip(hids, acts):
            conv.append(nn.Linear(in_channels, hid, bias=bias))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_channels = hid

        conv.append(nn.Linear(in_channels, out_channels, bias=bias))
        self.conv = Sequential(*conv)

        self.proj = nn.Linear(out_channels, 1)
        self.conv1d_1 = nn.Conv1d(out_channels, out_channels, kernel,
                                  padding=int((kernel - 1) / 2))
        self.conv1d_2 = nn.Conv1d(out_channels, out_channels, kernel,
                                  padding=int((kernel - 1) / 2))
        self.lin = nn.Linear(2 * out_channels, out_channels)
        self.conv1d_dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.proj.reset_parameters()
        self.conv1d_1.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index=None, edge_weight=None):
        """"""
        x1 = self.conv(x)
        g_score = self.proj(x1)  # [num_nodes, 1]
        g_score_sorted, sort_idx = torch.sort(g_score, dim=0)
        _, inverse_idx = torch.sort(sort_idx, dim=0)

        sorted_x = g_score_sorted * x1[sort_idx].squeeze()
        sorted_x = torch.transpose(sorted_x, 0, 1).unsqueeze(
            0)  # [1, dataset.num_classes, num_nodes]
        sorted_x = self.conv1d_1(sorted_x).relu()
        sorted_x = self.conv1d_dropout(sorted_x)
        sorted_x = self.conv1d_2(sorted_x)
        # [num_nodes, dataset.num_classes]
        sorted_x = torch.transpose(sorted_x.squeeze(), 0, 1)
        # [num_nodes, dataset.num_classes]
        x2 = sorted_x[inverse_idx].squeeze()

        out = torch.cat([x1, x2], dim=1)
        out = self.lin(out)
        return out


class NLGAT(nn.Module):
    r"""Non-Local Graph Neural Networks (NLGNN) with
    :class:`GAT` as backbone from the
    `"Non-Local Graph Neural Networks"
    <https://ieeexplore.ieee.org/document/9645300>`_ paper (TPAMI'22)

    Parameters
    ----------
    in_channels : int,
        the input dimensions of model
    out_channels : int,
        the output dimensions of model
    hids : List[int], optional
        the number of hidden units for each hidden layer, by default [8]
    num_heads : list, optional
        the number of attention heads for each hidden layer, by default [8]
    acts : List[str], optional
        the activation function for each hidden layer, by default ['relu']
    kernel : int,
        the number of kernel used in :class:`nn.Conv1d`, by default 5
    dropout : float, optional
        the dropout ratio of model, by default 0.6
    bias : bool, optional
        whether to use bias in the layers, by default True
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer,
        by default False



    Examples
    --------
    >>> # NLGAT with one hidden layer
    >>> model = NLGAT(100, 10)

    >>> # NLGAT with two hidden layers
    >>> model = NLGAT(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # NLGAT with two hidden layers, without first activation
    >>> model = NLGAT(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # NLGAT with deep architectures, each layer has elu activation
    >>> model = NLGAT(100, 10, hids=[16]*8, acts=['elu'])

    See also
    --------
    :class:`greatx.nn.models.supervised.NLGCN`
    :class:`greatx.nn.models.supervised.NLMLP`

    Reference:

    * https://github.com/divelab/Non-Local-GNN

    """
    @wrapper
    def __init__(self, in_channels: int, out_channels: int,
                 hids: List[int] = [8], num_heads: list = [8],
                 acts: List[str] = ['elu'], kernel: int = 5,
                 dropout: float = 0.6, bias: bool = True, bn: bool = False,
                 includes=['num_heads']):

        super().__init__()
        head = 1
        conv = []
        for hid, num_head, act in zip(hids, num_heads, acts):
            conv.append(
                GATConv(in_channels * head, hid, heads=num_head, bias=bias,
                        dropout=dropout))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_channels = hid
            head = num_head

        conv.append(
            GATConv(in_channels * head, out_channels, heads=1, bias=bias,
                    concat=False, dropout=dropout))
        self.conv = Sequential(*conv)

        self.proj = nn.Linear(out_channels, 1)
        self.conv1d_1 = nn.Conv1d(out_channels, out_channels, kernel,
                                  padding=int((kernel - 1) / 2))
        self.conv1d_2 = nn.Conv1d(out_channels, out_channels, kernel,
                                  padding=int((kernel - 1) / 2))
        self.lin = nn.Linear(2 * out_channels, out_channels)
        self.conv1d_dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.proj.reset_parameters()
        self.conv1d_1.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index=None, edge_weight=None):
        """"""
        x1 = self.conv(x, edge_index, edge_weight)
        g_score = self.proj(x1)  # [num_nodes, 1]
        g_score_sorted, sort_idx = torch.sort(g_score, dim=0)
        _, inverse_idx = torch.sort(sort_idx, dim=0)

        sorted_x = g_score_sorted * x1[sort_idx].squeeze()
        sorted_x = torch.transpose(sorted_x, 0, 1).unsqueeze(
            0)  # [1, dataset.num_classes, num_nodes]
        sorted_x = self.conv1d_1(sorted_x).relu()
        sorted_x = self.conv1d_dropout(sorted_x)
        sorted_x = self.conv1d_2(sorted_x)
        # [num_nodes, dataset.num_classes]
        sorted_x = torch.transpose(sorted_x.squeeze(), 0, 1)
        # [num_nodes, dataset.num_classes]
        x2 = sorted_x[inverse_idx].squeeze()

        out = torch.cat([x1, x2], dim=1)
        out = self.lin(out)
        return out
