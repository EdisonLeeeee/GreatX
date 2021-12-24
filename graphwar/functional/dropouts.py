import torch
from graphwar.functional.subgraph import subgraph

def drop_edge(g, p=0.5, training=True):
    """
    DropEdge: Sampling edge using a uniform distribution.
    """
    
    if not training or not p:
        return g

    g = g.local_var()
    num_edges = g.num_edges()
    num_drops = int(p * num_edges)
    perm = torch.randperm(num_edges, device=g.device)
    g.remove_edges(perm[:num_drops])
    return g


def drop_node(g, p=0.5, training=True):
    """
    DropNode: Sampling node using a uniform distribution.
    """
    
    if not training or not p:
        return g

    num_nodes = g.num_nodes()
    nodes = torch.arange(num_nodes, dtype=torch.long)
    mask = torch.full_like(nodes, 1 - p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    subset = nodes[mask]
    return subgraph(g, subset)