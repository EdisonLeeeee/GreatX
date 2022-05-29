from typing import Optional, Tuple

import torch
import torch_cluster
from torch import Tensor

from torch_geometric.utils import subgraph, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes


def drop_edge(edge_index: Tensor, edge_weight: Optional[Tensor] = None,
              p: float = 0.5, training: bool = True) -> Tuple[Tensor, Tensor]:
    """
    DropEdge: Sampling edge using a uniform distribution.
    """

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or not p:
        return edge_index, edge_weight

    num_edges = edge_index.size(1)
    e_ids = torch.arange(num_edges, dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    edge_index = edge_index[:, ~mask]
    if edge_weight is not None:
        edge_weight = edge_weight[:, ~mask]
    return edge_index, edge_weight


def drop_node(edge_index: Tensor, edge_weight: Optional[Tensor] = None,
              p: float = 0.5, training: bool = True,
              num_nodes: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """
    DropNode: Sampling node using a uniform distribution.
    """

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or not p:
        return edge_index, edge_weight

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    nodes = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(nodes, 1 - p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    subset = nodes[mask]
    return subgraph(subset, edge_index, edge_weight)


def drop_path(edge_index: Tensor, edge_weight: Optional[Tensor] = None,
              r: float = 0.5,
              walks_per_node: int = 2,
              walk_length: int = 4,
              p: float = 1, q: float = 1,
              training: bool = True,
              num_nodes: int = None,
              by: str = 'degree') -> Tuple[Tensor, Tensor]:

    if r < 0. or r > 1.:
        raise ValueError(f'Root node sampling ratio `r` has to be between 0 and 1 '
                         f'(got {r}')

    if not training or not r:
        return edge_index, edge_weight

    assert by in {'degree', 'uniform'}
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float)

    if isinstance(r, (int, float)):
        num_starts = int(r * num_nodes)
        if by == 'degree':
            prob = deg / deg.sum()
            start = prob.multinomial(num_samples=num_starts, replacement=True)
        else:
            start = torch.randperm(num_nodes, device=edge_index.device)[
                :num_starts]
    elif torch.is_tensor(r):
        start = r.to(edge_index)
    else:
        raise ValueError('Root node sampling ratio `r` must be '
                         f'`float`, `torch.Tensor`, but got {r}.')

    if walks_per_node:
        start = start.repeat(walks_per_node)

    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])

    n_id, e_id = torch.ops.torch_cluster.random_walk(
        rowptr, col, start, walk_length, p, q)
    mask = row.new_ones(row.size(0), dtype=torch.bool)
    mask[e_id.view(-1)] = False

    if edge_weight is not None:
        edge_weight = edge_weight[mask]
    return edge_index[:, mask], edge_weight
