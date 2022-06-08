from collections import namedtuple
from typing import Union

import numpy as np
import scipy.sparse as sp
from numba import njit, types
from numba.typed import Dict

ego_graph_nodes_edges = namedtuple('ego_graph', ['nodes', 'edges'])


__all__ = ['ego_graph']


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

    Note
    ----
    This is a faster implementation of 
    :class:`networkx.ego_graph` based on scipy sparse matrix and numba


    See Also
    --------
    :class:`networkx.ego_graph`
    :class:`torch_geometric.utils.k_hop_subgraph`

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
        e = _get_remaining_edges(
            indices, indptr, np.array(targets[start:]), seen, hops)
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
