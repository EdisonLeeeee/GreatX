import torch

def drop_edge(g, p=0.5, training=True):
    if not training or not p:
        return g

    g = g.local_var()
    num_edges = g.num_edges()
    num_drops = int(p * num_edges)
    perm = torch.randperm(num_edges, device=g.device)
    g.remove_edges(perm[:num_drops])
    return g
