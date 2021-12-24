import torch

def subgraph(g, subset):
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