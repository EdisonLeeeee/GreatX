import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import DGLError
from graphwar.utils.normalize import dgl_normalize
from graphwar.functional import spmm
from typing import Optional

try:
    from glcore import dimmedian_idx
except (ModuleNotFoundError, ImportError):
    dimmedian_idx = None

try:
    from glcore import topk
except (ModuleNotFoundError, ImportError):
    topk = None


class DimwiseMedianConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 add_self_loop=True,
                 row_normalize=False,
                 norm='none',
                 activation=None,
                 weight=True,
                 bias=True):

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
        
        if self._add_self_loop:
            graph = graph.add_self_loop()
            if edge_weight is not None:
                size = (graph.num_nodes(),) + edge_weight.size()[1:]
                self_loop = edge_weight.new_ones(size)
                edge_weight = torch.cat([edge_weight, self_loop])
        else:
            graph = graph.local_var()

        edge_weight = dgl_normalize(graph, self._norm, edge_weight)

        if self.weight is not None:
            feat = feat @ self.weight

        # ========= weighted dimension-wise Median aggregation ===
        N, D = feat.size()
        row, col, e_id = graph.edges(order='srcdst', form='all')
        edge_index = torch.stack([row, col], dim=0)
        
        if self._norm != 'none':
            # if edge_weight is all 1 and it is not necessary
            # to sort again
            edge_weight = edge_weight[e_id]
            
        median_idx = dimmedian_idx(feat, edge_index, edge_weight, N)
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


class SoftKConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 add_self_loop=True,
                 row_normalize=False,
                 k=32,
                 temperature=1.0,
                 with_weight_correction=True,
                 norm='none',
                 activation=None,
                 weight=True,
                 bias=True):

        super().__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))

        if topk is None:
            raise RuntimeWarning("Module 'glcore' is not properly installed, please refer to "
                                 "'https://github.com/EdisonLeeeee/glcore' for more information.")

        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._add_self_loop = add_self_loop
        self._row_normalize = row_normalize
        self._k = k
        self._temperature = temperature
        self._with_weight_correction = with_weight_correction
        self._activation = activation

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
        Soft Weighted Medoid topk aggregation.

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

        if self._add_self_loop:
            graph = graph.add_self_loop()
            if edge_weight is not None:
                size = (graph.num_nodes(),) + edge_weight.size()[1:]
                self_loop = edge_weight.new_ones(size)
                edge_weight = torch.cat([edge_weight, self_loop])
        else:
            graph = graph.local_var()
        edge_weight = dgl_normalize(graph, self._norm, edge_weight)

        if self.weight is not None:
            feat = feat @ self.weight

        # ========= Soft Weighted Medoid in the top `k` neighborhood ===
        feat = soft_weighted_medoid_k_neighborhood(graph, feat, edge_weight, k=self._k,
                                                   temperature=self._temperature,
                                                   with_weight_correction=self._with_weight_correction,
                                                   row_normalize=self._row_normalize)
        # ==============================================================

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


def soft_weighted_medoid_k_neighborhood(
    g: dgl.DGLGraph,
    feat: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    k: int = 32,
    temperature: float = 1.0,
    with_weight_correction: bool = True,
    row_normalize: bool = False
) -> torch.Tensor:
    """Soft Weighted Medoid in the top `k` neighborhood (see Eq. 6 and Eq. 7 in our paper). 
    This function can be used as a robust aggregation function 
    within a message passing GNN (e.g. see `models#RGNN`).

    Note that if `with_weight_correction` is false, 
    we calculate the Weighted Soft Medoid as in Appendix C.4.

    Parameters
    ----------
    g : dgl.DGLGraph
        dgl graph instance.
    x : torch.Tensor
        Dense [n, d] tensor containing the node attributes/embeddings.
    edge_weight : torch.Tensor, optional
        edge weights of the edges in the graph `g`, by default `None` (1 for all edges).
    k : int, optional
        Neighborhood size for selecting the top k elements, by default 32.
    temperature : float, optional
        Controlling the steepness of the softmax, by default 1.0.
    with_weight_correction : bool, optional
        For enabling an alternative normalisazion (see above), by default True.
    row_normalize : bool, optional
        whether to perform normalization for aggregated features, by default False.        

    Returns
    -------
    torch.Tensor
        The new embeddings [n, d]   
    """

    n = feat.size(0)
    assert k <= n

    row, col, e_id = g.edges(order='srcdst', form='all')
    if edge_weight is None:
        edge_weight = row.new_ones(row.size(0), dtype=torch.float)
    else:
        edge_weight = edge_weight[e_id]

    edge_index = torch.stack([row, col], dim=0)

    # Custom CUDA extension code for the top k values of the sparse adjacency matrix
    top_k_weights, top_k_idx = topk(edge_index, edge_weight, n, k)

    # Partial distance matrix calculation
    distances_top_k = partial_distance_matrix(feat, top_k_idx)

    # Multiply distances with weights
    distances_top_k = (top_k_weights[:, None, :].expand(n, k, k) * distances_top_k).sum(-1)
    distances_top_k[top_k_idx == -1] = torch.finfo(distances_top_k.dtype).max
    distances_top_k[~torch.isfinite(distances_top_k)] = torch.finfo(distances_top_k.dtype).max

    # Softmax over L1 criterium
    reliable_edge_weight = F.softmax(-distances_top_k / temperature, dim=-1)
    del distances_top_k

    # To have GCN as a special case (see Eq. 6 in our paper)
    if with_weight_correction:
        reliable_edge_weight = reliable_edge_weight * top_k_weights
        reliable_edge_weight = reliable_edge_weight / reliable_edge_weight.sum(-1).view(-1, 1)

    # Map the top k results back to the (sparse) [n,n] matrix
    top_k_inv_idx_row = torch.arange(n, device=g.device)[:, None].expand(n, k).flatten()
    top_k_inv_idx_column = top_k_idx.flatten()
    top_k_mask = top_k_inv_idx_column != -1

    # Note: The adjacency matrix A might have disconnected nodes. In that case applying the top_k_mask will
    # drop the nodes completely from the adj matrix making, changing its shape
    reliable_edge_index = torch.stack([top_k_inv_idx_row[top_k_mask], top_k_inv_idx_column[top_k_mask]])
    reliable_edge_weight = reliable_edge_weight[top_k_mask.view(n, k)]

    out = spmm(reliable_edge_index, reliable_edge_weight, n, feat)
    # Normalization and calculation of new embeddings
    if row_normalize:
        row_sum = edge_weight.new_zeros(feat.size(0))
        row_sum.scatter_add_(0, row, edge_weight)
        out = row_sum.view(-1, 1) * out
    return out


def partial_distance_matrix(feat: torch.Tensor, partial_idx: torch.Tensor) -> torch.Tensor:
    """Calculates the partial distance matrix given the indices. 
    For a low memory footprint (small computation graph)
    it is essential to avoid duplicated computation of the distances.

    Parameters
    ----------
    x : torch.Tensor
        Dense [n, d] tensor with attributes to calculate the distance between.
    partial_idx : torch.Tensor
        Dense [n, k] tensor where `-1` stands for no index.
        Pairs are generated by the row id and the contained ids.

    Returns
    -------
    torch.Tensor
        [n, k, k] distances matrix (zero entries for `-1` indices)    
    """
    n, k = partial_idx.size()

    # Permute the indices of partial_idx
    idx_row = partial_idx[:, None, :].expand(n, k, k).flatten()
    idx_column = partial_idx[:, None, :].expand(n, k, k).transpose(1, 2).flatten()
    is_not_missing_mask = (idx_row != -1) & (idx_column != -1)
    idx_row, idx_column = idx_row[is_not_missing_mask], idx_column[is_not_missing_mask]

    # Use symmetry of Euclidean distance to half memory footprint
    symmetry_mask = idx_column < idx_row
    idx_row[symmetry_mask], idx_column[symmetry_mask] = idx_column[symmetry_mask], idx_row[symmetry_mask]
    del symmetry_mask

    # Create linear index (faster deduplication)
    linear_index = idx_row * n + idx_column
    del idx_row
    del idx_column

    # Avoid duplicated distance calculation (helps greatly for space cost of backward)
    distance_matrix_idx, unique_reverse_index = torch.unique(linear_index, return_inverse=True)

    # Calculate Euclidean distances between all pairs
    sparse_distances = torch.norm(feat[torch.div(distance_matrix_idx, n, rounding_mode='floor')] - feat[distance_matrix_idx % n], dim=1)

    # Create dense output
    out = torch.zeros(n * k * k, dtype=torch.float, device=feat.device)

    # Map sparse distances to output tensor
    out[is_not_missing_mask] = sparse_distances[unique_reverse_index]

    return out.view(n, k, k)
