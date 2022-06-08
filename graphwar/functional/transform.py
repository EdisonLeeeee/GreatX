from typing import Optional
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes

__all__ = ['to_sparse_tensor', 'to_dense_adj']


def to_sparse_tensor(edge_index: Tensor, edge_weight: Optional[Tensor] = None,
                     num_nodes: Optional[int] = None, is_sorted: bool = False) -> SparseTensor:
    """Convert edge index to a torch_sparse sparse tensor

    Parameters
    ----------
    edge_index : Tensor
        edge index with shape [2, M]
    edge_weight : Optional[Tensor], optional
        edge weight with shape [M], by default None
    num_nodes : Optional[int], optional
        the number of nodes in the graph, by default None
    is_sorted : bool, optional
        whether the :obj:`edge_index` is sorted, by default False

    Returns
    -------
    SparseTensor
        the output sparse adjacency matrix denoted as torch_sparse.SparseTensor,
        with shape [num_nodes, num_nodes]
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    return SparseTensor.from_edge_index(
        edge_index, edge_weight, is_sorted=is_sorted,
        sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)


def to_dense_adj(edge_index: Tensor, edge_weight: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, fill_value: float = 1.0) -> Tensor:
    """Convert edge index to dense adjacency matrix

    Parameters
    ----------
    edge_index : Tensor
        edge index with shape [2, M]
    edge_weight : Optional[Tensor], optional
        edge weight with shape [M], by default None
    num_nodes : Optional[int], optional
        the number of nodes in the graph, by default None
    fill_value : float
        filling value for elements in the adjacency matrix 
        where edges existed, by default 1.0

    Returns
    -------
    Tensor
        output dense adjacency matrix with shape :obj:`[num_nodes, num_nodes]`
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    if edge_weight is None:
        adj[edge_index[0], edge_index[1]] = fill_value
    else:
        adj[edge_index[0], edge_index[1]] = edge_weight
    return adj
