import torch
import dgl
import numpy as np
import scipy.sparse as sp

from torch import Tensor
from numba import njit
from numba import types
from numba.typed import Dict

from typing import Union, Tuple, Optional
from collections import namedtuple

ego_graph_nodes_edges = namedtuple('ego_graph', ['nodes', 'edges'])


__all__ = ['maybe_num_nodes', 'k_hop_subgraph', 'ego_graph', 'subgraph']


def maybe_num_nodes(edge_index: Tensor,
                    num_nodes: Optional[int] = None) -> int:
    if num_nodes is not None:
        return num_nodes
    else:
        return edge_index.max().item() + 1 if edge_index.numel() > 0 else 0


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target') -> Tuple[Tensor, Tensor, Tensor]:
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns:
    (1) the nodes involved in the subgraph
    (2) the filtered :obj:`edge_index` connectivity
    (3) the edge mask indicating which edges were preserved.

    Parameters
    ----------
    node_idx (int, list, tuple or :obj:`Tensor`)
        The central node(s).
    num_hops: (int): 
        The number of hops :math:`k`.
    edge_index (LongTensor): 
        The edge indices.
    relabel_nodes (bool, optional): 
        If set to :obj:`True`, the resulting
        :obj:`edge_index` will be relabeled to hold consecutive indices
        starting from zero. (default: :obj:`False`)
    num_nodes (int, optional): 
        The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    flow (string, optional): 
        The flow direction of :math:`k`-hop
        aggregation (:obj:`"source_to_target"` or
        :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    Returns
    -------
    subset nodes, subset edge index and edge mask.


    See Also
    --------
    ego_graph
    """
    if isinstance(edge_index, tuple):
        edge_index = torch.stack(edge_index, dim=0)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset = torch.cat(subsets).unique()

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, edge_mask


def ego_graph(adj_matrix: sp.csr_matrix,
              targets: Union[int, list], hops: int = 1) -> ego_graph_nodes_edges:
    """Returns induced subgraph of neighbors centered at node n within
    a given radius.

    Parameters
    ----------
    adj_matrix : sp.csr_matrix,
        a Scipy CSR sparse adjacency matrix representing a graph

    targets : Union[int, list]
        center nodes, a single node or a list of nodes

    hops : int number, optional
        Include all neighbors of distance<=hops from nodes.

    Returns
    -------
    NamedTuple(nodes, edges):
        nodes: shape [N], the nodes of the subgraph
        edges: shape [2, M], the edges of the subgraph

    Notes
    -----
    This is a faster implementation of 
    `networkx.ego_graph` based on scipy sparse matrix and numba


    See Also
    --------
    networkx.ego_graph
    k_hop_subgraph

    """
    assert sp.issparse(adj_matrix)
    adj_matrix = adj_matrix.tocsr(copy=False)

    if np.ndim(targets) == 0:
        targets = [targets]
    elif isinstance(targets, np.ndarray):
        targets = targets.tolist()
    else:
        targets = list(targets)

    indices = adj_matrix.indices
    indptr = adj_matrix.indptr

    edges = {}
    start = 0
    N = adj_matrix.shape[0]
    seen = np.zeros(N) - 1
    seen[targets] = 0
    for level in range(hops):
        end = len(targets)
        while start < end:
            head = targets[start]
            nbrs = indices[indptr[head]:indptr[head + 1]]
            for u in nbrs:
                if seen[u] < 0:
                    targets.append(u)
                    seen[u] = level + 1
                if (u, head) not in edges:
                    edges[(head, u)] = level + 1

            start += 1

    if len(targets[start:]):
        e = _get_remaining_edges(indices, indptr, np.array(targets[start:]), seen, hops)
    else:
        e = []

    return ego_graph_nodes_edges(nodes=np.asarray(targets),
                                 edges=np.asarray(list(edges.keys()) + e).T)


@njit
def _get_remaining_edges(indices: np.ndarray, indptr: np.ndarray,
                         last_level: np.ndarray, seen: np.ndarray,
                         hops: int) -> list:
    edges = []
    mapping = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )
    for u in last_level:
        nbrs = indices[indptr[u]:indptr[u + 1]]
        nbrs = nbrs[seen[nbrs] == hops]
        mapping[u] = 1
        for v in nbrs:
            if not v in mapping:
                edges.append((u, v))
    return edges


def subgraph(g: dgl.DGLGraph,
             subset: Union[Tensor, int]) -> dgl.DGLGraph:
    """Returns the induced subgraph of :obj:`g`
    containing the nodes in :obj:`subset`.

    """
    g = g.local_var()
    row, col = g.edges()
    device = g.device
    num_nodes = g.num_nodes()

    node_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    node_mask[subset] = 0
    # mask to remove
    edge_mask = node_mask[row] | node_mask[col]
    e_ids = edge_mask.nonzero().view(-1)
    g.remove_edges(e_ids)
    return g
