from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor
from torch_geometric.utils import coalesce

from graphwar.nn.layers import GCNConv, activations
from graphwar.utils import wrapper


class SimPGCN(nn.Module):
    r"""Similarity Preserving Graph Convolution Network (SimPGCN)
    from the `"Node Similarity Preserving Graph Convolutional Networks"
    <https://arxiv.org/abs/2011.09643>`_ paper (WSDM'21)

    Parameters
    ----------
    in_channels : int, 
        the input dimensions of model
    out_channels : int, 
        the output dimensions of model
    hids : list, optional
        the number of hidden units for each hidden layer, by default [64]
    acts : list, optional
        the activation function for each hidden layer, by default None
    dropout : float, optional
        the dropout ratio of model, by default 0.5
    bias : bool, optional
        whether to use bias in the layers, by default True
    gamma : float, optional
        trade-off hyperparameter, by default 0.01
    bn: bool, optional (*NOT IMPLEMENTED NOW*)
        whether to use :class:`BatchNorm1d` after the convolution layer, by default False         

    Note
    ----
    It is convenient to extend the number of layers with different or the same
    hidden units (activation functions) using :meth:`graphwar.utils.wrapper`. 

    See Examples below:

    Examples
    --------
    >>> # SimPGCN with one hidden layer
    >>> model = SimPGCN(100, 10)

    >>> # SimPGCN with two hidden layers
    >>> model = SimPGCN(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # SimPGCN with two hidden layers, without activation at the first layer
    >>> model = SimPGCN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # SimPGCN with very deep architectures, each layer has elu as activation function
    >>> model = SimPGCN(100, 10, hids=[16]*8, acts=['elu'])

    """
    @wrapper
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hids: list = [64],
                 acts: list = [None],
                 dropout: float = 0.5,
                 bias: bool = True,
                 gamma: float = 0.01,
                 bn: bool = False  # TODO
                 ):

        super().__init__()

        if bn:
            raise NotImplementedError

        assert bias == True

        layers = nn.ModuleList()
        act_layers = nn.ModuleList()

        inc = in_channels
        for hid, act in zip(hids, acts):
            layers.append(GCNConv(in_channels,
                                  hid,
                                  bias=bias))
            act_layers.append(activations.get(act))
            inc = hid

        layers.append(GCNConv(inc,
                              out_channels,
                              bias=bias))
        act_layers.append(activations.get(None))

        self.layers = layers
        self.act_layers = act_layers
        self.scores = nn.ParameterList()
        self.bias = nn.ParameterList()
        self.D_k = nn.ParameterList()
        self.D_bias = nn.ParameterList()

        for hid in [in_channels] + hids:
            self.scores.append(nn.Parameter(torch.FloatTensor(hid, 1)))
            self.bias.append(nn.Parameter(torch.FloatTensor(1)))
            self.D_k.append(nn.Parameter(torch.FloatTensor(hid, 1)))
            self.D_bias.append(nn.Parameter(torch.FloatTensor(1)))

        # discriminator for ssl
        self.linear = nn.Linear(hids[-1], 1)
        self.dropout = nn.Dropout(dropout)
        self.gamma = gamma
        self.reset_parameters()

    def reset_parameters(self):

        for layer in self.layers:
            layer.reset_parameters()

        for s in self.scores:
            nn.init.xavier_uniform_(s)

        for bias in self.bias:
            # fill in b with positive value to make
            # score s closer to 1 at the beginning
            zeros(bias)

        for Dk in self.D_k:
            nn.init.xavier_uniform_(Dk)

        for bias in self.D_bias:
            zeros(bias)

        self.cache_clear()

    def cache_clear(self):
        """Clear cached inputs or intermediate results."""
        self._adj_knn = self._pseudo_labels = self._node_pairs = None
        return self

    def forward(self, x, edge_index, edge_weight=None):

        if self._adj_knn is None:
            self._adj_knn = adj_knn = knn_graph(x)
            # save for training
            self._pseudo_labels, self._node_pairs = attr_sim(x)
        else:
            adj_knn = self._adj_knn

        gamma = self.gamma
        embedding = None
        for ix, (layer, act) in enumerate(zip(self.layers, self.act_layers)):
            s = torch.sigmoid(x @ self.scores[ix] + self.bias[ix])

            Dk = x @ self.D_k[ix] + self.D_bias[ix]

            # graph convolution without graph structure
            tmp = layer.lin(x)
            if layer.bias is not None:
                tmp = tmp + layer.bias

            # adj_knn does not need to add self-loop edges
#             add_self_loops = layer.add_self_loops
#             layer.add_self_loops = False
            tmp_knn = layer(x, adj_knn)
#             layer.add_self_loops = add_self_loops

            # taken together
            x = s * act(layer(x, edge_index, edge_weight)) + (1 - s) * \
                act(tmp_knn) + gamma * Dk * act(tmp)

            if ix < len(self.layers) - 1:
                x = self.dropout(x)

            if ix == len(self.layers) - 2:
                embedding = x

        if self.training:
            return x, embedding
        else:
            return x

    def regression_loss(self, embeddings):
        K = 10000
        node_pairs = self._node_pairs
        pseudo_labels = self._pseudo_labels
        if len(node_pairs[0]) > K:
            #             sampled = np.random.choice(len(node_pairs[0]), K, replace=False)
            prob = torch.full((len(node_pairs[0]),), 1./len(node_pairs[0]))
            sampled = prob.multinomial(num_samples=K, replacement=False)

            embeddings0 = embeddings[node_pairs[0][sampled]]
            embeddings1 = embeddings[node_pairs[1][sampled]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(
                embeddings, pseudo_labels[sampled].unsqueeze(-1), reduction='mean')
        else:
            embeddings0 = embeddings[node_pairs[0]]
            embeddings1 = embeddings[node_pairs[1]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(
                embeddings, pseudo_labels.unsqueeze(-1), reduction='mean')
        return loss


def knn_graph(x: torch.Tensor, k: int = 20) -> SparseTensor:
    """Return a K-NN graph based on cosine similarity.

    """
    x = x.bool().float()  # x[x!=0] = 1
    sims = pairwise_cosine_similarity(x)
    sims = sims - torch.diag(torch.diag(sims))  # remove self-loops

    row = torch.arange(x.size(0), device=x.device).repeat_interleave(k)
    topk = torch.topk(sims, k=k, dim=1)
    col = topk.indices.flatten()
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = topk.values.flatten()

    N = x.size(0)
    adj = SparseTensor.from_edge_index(
        edge_index, edge_weight, sparse_sizes=(N, N))

    return adj


def pairwise_cosine_similarity(X: torch.Tensor,
                               Y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute cosine similarity between samples in X and Y.

    Cosine similarity, or the cosine kernel, computes similarity as the
    normalized dot product of X and Y:

        K(X, Y) = <X, Y> / (||X||*||Y||)

    On L2-normalized data, this function is equivalent to linear_kernel.


    Parameters
    ----------
    X : torch.Tensor, shape (N, M)
        Input data.
    Y : Optional[torch.Tensor], optional
        Input data. If ``None``, the output will be the pairwise
        similarities between all samples in ``X``., by default None

    Returns
    -------
    torch.Tensor, shape (N, M)
        the pairwise similarities matrix
    """

    A_norm = X / X.norm(dim=1)[:, None]
    if Y is None:
        B_norm = A_norm
    else:
        B_norm = Y / Y.norm(dim=1)[:, None]
    S = torch.mm(A_norm, B_norm.transpose(0, 1))
    return S


def attr_sim(x, k=5):
    x = x.bool().float()  # x[x!=0] = 1

    sims = pairwise_cosine_similarity(x)
    indices_sorted = sims.argsort(1)
    selected = torch.cat((indices_sorted[:, :k],
                          indices_sorted[:, - k - 1:]), dim=1)
    row = torch.arange(x.size(0), device=x.device).repeat_interleave(
        selected.size(1))
    col = selected.view(-1)

    mask = row != col
    row, col = row[mask], col[mask]

    mask = row > col
    row[mask], col[mask] = col[mask].clone(), row[mask].clone()

    node_pairs = torch.stack([row, col], dim=0)
    node_pairs = coalesce(node_pairs, num_nodes=x.size(0))
    return sims[node_pairs[0], node_pairs[1]], node_pairs
