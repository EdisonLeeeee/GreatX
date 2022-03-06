from typing import Optional

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLError

from graphwar.functional import spmm
from graphwar.functional.transform import dgl_normalize

try:
    from glcore import dimmedian_idx
except (ModuleNotFoundError, ImportError):
    dimmedian_idx = None


class DimwiseMedianConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 add_self_loop=True,
                 row_normalize=False,
                 norm='none',
                 activation=None,
                 weight=True,
                 bias=True,
                 cached=True):

        super().__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))

        if dimmedian_idx is None:
            raise RuntimeWarning("Module 'glcore' is not properly installed, please refer to "
                                 "'https://github.com/EdisonLeeeee/glcore' for more information.")

        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._add_self_loop = add_self_loop
        self._row_normalize = row_normalize
        self._activation = activation
        self._cached = cached
        self._cached_edges = None

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute Graph Convolution layer with
        Weighted Medoid aggregation.


        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        edge_weight : torch.Tensor, optional
            Optional edge weight for each edge.            

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """

        assert edge_weight is None or edge_weight.size(0) == graph.num_edges()

        if self._cached and self._cached_edges is not None:
            row, col, edge_weight = self._cached_edges
        else:
            if self._add_self_loop:
                graph = graph.add_self_loop()
                if edge_weight is not None:
                    size = (graph.num_nodes(),) + edge_weight.size()[1:]
                    self_loop = edge_weight.new_ones(size)
                    edge_weight = torch.cat([edge_weight, self_loop])
            else:
                graph = graph.local_var()

            edge_weight = dgl_normalize(graph, self._norm, edge_weight)

            row, col, e_id = graph.edges(order='srcdst', form='all')
            edge_weight = edge_weight[e_id]

            # cache edges
            if self._cached:
                self._cached_edges = row, col, edge_weight

        if self.weight is not None:
            feat = feat @ self.weight

        # ========= weighted dimension-wise Median aggregation ===
        N, D = feat.size()
        median_idx = dimmedian_idx(feat, row, col, edge_weight, N)
        col_idx = torch.arange(D, device=graph.device).view(1, -1).expand(N, D)
        feat = feat[median_idx, col_idx]
        # Normalization and calculation of new embeddings
        if self._row_normalize:
            row_sum = edge_weight.new_zeros(feat.size(0))
            row_sum.scatter_add_(0, row, edge_weight)
            feat = row_sum.view(-1, 1) * feat
        # ========================================================

        if self.bias is not None:
            feat = feat + self.bias

        if self._activation is not None:
            feat = self._activation(feat)
        return feat

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)
