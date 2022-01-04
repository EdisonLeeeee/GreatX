import torch
from dgl import DGLGraph

from graphwar.functional.subgraph import subgraph


def drop_edge(g: DGLGraph, p: float = 0.5, training: bool = True) -> DGLGraph:
    """
    DropEdge: Sampling edge using a uniform distribution.
    """

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or not p:
        return g

    g = g.local_var()
    num_edges = g.num_edges()
    e_ids = torch.arange(num_edges, dtype=torch.long, device=g.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    g.remove_edges(e_ids[mask])
    return g


def drop_node(g: DGLGraph, p: float = 0.5, training: bool = True) -> DGLGraph:
    """
    DropNode: Sampling node using a uniform distribution.
    """

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or not p:
        return g

    num_nodes = g.num_nodes()
    nodes = torch.arange(num_nodes, dtype=torch.long, device=g.device)
    mask = torch.full_like(nodes, 1 - p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    subset = nodes[mask]
    return subgraph(g, subset)
