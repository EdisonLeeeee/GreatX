import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from graphwar.config import Config
from graphwar.nn import GCNConv, activations
from graphwar.utils import wrapper
from graphwar.functional import knn_graph, attr_sim

_EDGE_WEIGHT = Config.edge_weight


class SimPGCN(nn.Module):
    """Similarity Preserving Graph Convolution Network (SimPGCN). 

    Example
    -------
    # SimPGCN with one hidden layer
    >>> model = SimPGCN(100, 10)
    # SimPGCN with two hidden layers
    >>> model = SimPGCN(100, 10, hids=[32, 16], acts=['relu', 'elu'])
    # SimPGCN with two hidden layers, without activation at the first layer
    >>> model = SimPGCN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    """
    @wrapper
    def __init__(self,
                 in_feats: int,
                 out_feats: int,
                 hids: list = [64],
                 acts: list = [None],
                 dropout: float = 0.5,
                 bn: bool = False,
                 bias: bool = False,
                 norm: str = 'both',
                 gamma: float = 0.1):

        super().__init__()
        assert len(hids) > 0

        layers = nn.ModuleList()
        act_layers = nn.ModuleList()

        inc = in_feats
        for hid, act in zip(hids, acts):
            layers.append(GCNConv(in_feats,
                                  hid,
                                  bias=bias, norm=norm))
            act_layers.append(activations.get(act))
            inc = hid

        layers.append(GCNConv(inc,
                              out_feats,
                              bias=bias, norm=norm))
        act_layers.append(activations.get(None))

        self.layers = layers
        self.act_layers = act_layers
        self.scores = nn.ParameterList()
        self.bias = nn.ParameterList()
        self.D_k = nn.ParameterList()
        self.D_bias = nn.ParameterList()

        for hid in [in_feats] + hids:
            self.scores.append(nn.Parameter(torch.FloatTensor(hid, 1)))
            self.bias.append(nn.Parameter(torch.FloatTensor(1)))
            self.D_k.append(nn.Parameter(torch.FloatTensor(hid, 1)))
            self.D_bias.append(nn.Parameter(torch.FloatTensor(1)))

        # discriminator for ssl
        self.linear = nn.Linear(hids[-1], 1)
        self.dropout = nn.Dropout(dropout)
        self.gamma = gamma
        self.cache_clear()
        self.reset_parameters()

    def reset_parameters(self):

        for layer in self.layers:
            layer.reset_parameters()

        for s in self.scores:
            nn.init.xavier_uniform_(s)

        for bias in self.bias:
            # fill in b with positive value to make
            # score s closer to 1 at the beginning
            nn.init.zeros_(bias)

        for Dk in self.D_k:
            nn.init.xavier_uniform_(Dk)

        for bias in self.D_bias:
            nn.init.zeros_(bias)

    def forward(self, g, feat, edge_weight=None):

        if self._g_knn is None:
            self._g_knn = g_knn = knn_graph(feat)
            # save for training
            self._pseudo_labels, self._node_pairs = attr_sim(feat)
        else:
            g_knn = self._g_knn

        if edge_weight is None:
            edge_weight = g.edata.get(_EDGE_WEIGHT, edge_weight)

        gamma = self.gamma
        embedding = None
        for ix, (layer, act) in enumerate(zip(self.layers, self.act_layers)):
            s = torch.sigmoid(feat @ self.scores[ix] + self.bias[ix])
            Dk = feat @ self.D_k[ix] + self.D_bias[ix]

            # graph convolution without graph structure
            tmp = feat @ layer.weight
            if layer.bias is not None:
                tmp = tmp + layer.bias

            # g_knn does not need to add selfloops
            _add_self_loop = layer._add_self_loop
            layer._add_self_loop = False
            tmp_knn = layer(g_knn, feat)
            layer._add_self_loop = _add_self_loop

            # taken together
            feat = s * act(layer(g, feat)) + (1 - s) * \
                act(tmp_knn) + gamma * Dk * act(tmp)

            if ix < len(self.layers) - 1:
                feat = self.dropout(feat)

            if ix == len(self.layers) - 2:
                embedding = feat

        if self.training:
            return feat, embedding
        else:
            return feat

    def regression_loss(self, embeddings):
        k = 10000
        node_pairs = self._node_pairs
        pseudo_labels = self._pseudo_labels
        if len(node_pairs[0]) > k:
            sampled = np.random.choice(len(node_pairs[0]), k, replace=False)

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

    def cache_clear(self):
        self._g_knn = self._pseudo_labels = self._node_pairs = None
        return self
