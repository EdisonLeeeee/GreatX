from typing import Optional

import torch
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

__all__ = ['to_sparse_tensor', 'to_dense_adj', 'to_sparse_adj']


def to_sparse_tensor(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    is_sorted: bool = False,
) -> SparseTensor:
    """Convert edge index to a :class:`torch_sparse.SparseTensor`

    Parameters
    ----------
    edge_index : torch.Tensor
        edge index with shape [2, M]
    edge_weight : Optional[Tensor], optional
        edge weight with shape [M], by default None
    num_nodes : Optional[int], optional
        the number of nodes in the graph, by default None
    is_sorted : bool, optional
        whether the :obj:`edge_index` is sorted, by default False

    Returns
    -------
    :class:`torch_sparse.SparseTensor`
        the output sparse adjacency matrix denoted as
        :class:`torch_sparse.SparseTensor`,
        with shape :obj:`[num_nodes, num_nodes]`
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    return SparseTensor.from_edge_index(
        edge_index, edge_weight, is_sorted=is_sorted,
        sparse_sizes=(num_nodes, num_nodes)).to(edge_index.device)


def to_dense_adj(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    fill_value: float = 1.0,
) -> Tensor:
    """Convert edge index to dense adjacency matrix :class:`torch.FloatTensor`

    Parameters
    ----------
    edge_index : torch.Tensor
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
    :class:`torch.Tensor`
        output dense adjacency matrix with shape :obj:`[num_nodes, num_nodes]`
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    if edge_weight is None:
        adj[edge_index[0], edge_index[1]] = fill_value
    else:
        adj[edge_index[0], edge_index[1]] = edge_weight
    return adj


def to_sparse_adj(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[int] = None,
) -> torch.sparse.Tensor:
    """Convert edge index to sparse adjacency matrix
    :class:`torch.sparse.Tensor`

    Parameters
    ----------
    edge_index : torch.Tensor
        edge index with shape [2, M]
    edge_attr : Optional[Tensor], optional
        edge attributes, by default None
    size : Optional[int], optional
        The size of the sparse matrix.
            If given as an integer, will create a quadratic sparse matrix.
            If set to :obj:`None`, will infer a quadratic sparse matrix based
            on :obj:`edge_index.max() + 1`. By default None

    Return
    -------
    :class:`torch.sparse.Tensor`
        output sparse adjacency matrix

    Example:
    -------

    >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
    ...                            [1, 0, 2, 1, 3, 2]])
    >>> to_torch_coo_tensor(edge_index)
    tensor(indices=tensor([[0, 1, 1, 2, 2, 3],
                        [1, 0, 2, 1, 3, 2]]),
        values=tensor([1., 1., 1., 1., 1., 1.]),
        size=(4, 4), nnz=6, layout=torch.sparse_coo)
    """

    if size is None:
        size = int(edge_index.max()) + 1
    if not isinstance(size, (tuple, list)):
        size = (size, size)

    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    size = tuple(size) + edge_attr.size()[1:]
    out = torch.sparse_coo_tensor(edge_index, edge_attr, size,
                                  device=edge_index.device)
    out = out.coalesce()
    return out
