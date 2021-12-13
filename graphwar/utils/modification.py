import dgl
import torch


def add_edges(g: dgl.DGLGraph, edges: torch.Tensor,
              symmetric=True) -> dgl.DGLGraph:
    """add edges to the graph `g`. This method is 
    similar to `DGLGraph.add_edges()` but returns a 
    copy of the graph `g`.

    Parameters
    ----------
    g : dgl.DGLGraph
        the dgl graph instance where edges will be added to.
    edges : torch.Tensor
        shape [2, M], the edges to be added in the graph.
    symmetric : bool
        whether the graph is symmetric, if True,
        it would flip the edges in the graph by:
        `edges = torch.cat([edges, edges[[1,0]]])`

    Returns
    -------
    dgl.DGLGraph
        the dgl graph instance with edge added.
    """

    edges = edges.to(g.device)

    if symmetric:
        edges = torch.cat([edges, edges[[1, 0]]], dim=1)

    g = g.local_var()
    g.add_edges(edges[0], edges[1])
    return g


def remove_edges(g: dgl.DGLGraph, edges: torch.Tensor,
                 symmetric=True) -> dgl.DGLGraph:
    """remove edges from the graph `g`. 

    Parameters
    ----------
    g : dgl.DGLGraph
        the dgl graph instance where edges will be removed from.
    edges : torch.Tensor
        shape [2, M], the edges to be removed in the graph.
    symmetric : bool
        whether the graph is symmetric, if True,
        it would flip the edges in the graph by:
        `edges = torch.cat([edges, edges[[1,0]]])`

    Returns
    -------
    dgl.DGLGraph
        the dgl graph instance with edge removed.
    """

    edges = edges.to(g.device)

    if symmetric:
        edges = torch.cat([edges, edges[[1, 0]]], dim=1)
    g = g.local_var()
    row, col = edges

    mask = g.has_edges_between(row, col)

    row_to_remove = row[mask]
    col_to_remove = col[mask]

    eids = g.edge_ids(row_to_remove, col_to_remove)
    g.remove_edges(eids)

    return g


def flip_graph(g: dgl.DGLGraph, edges: torch.Tensor,
               symmetric=True) -> dgl.DGLGraph:
    """flip edges in the graph `g`

    Parameters
    ----------
    g : dgl.DGLGraph
        the dgl graph instance where edges will be flipped from.
    edges : torch.Tensor
        shape [2, M], the edges to be flipped in the graph.
    symmetric : bool
        whether the graph is symmetric, if True,
        it would flip the edges in the graph by:
        `edges = torch.cat([edges, edges[[1,0]]])`

    Returns
    -------
    dgl.DGLGraph
        the dgl graph instance with edge flipped.
    """
    edges = edges.to(g.device)

    if symmetric:
        edges = torch.cat([edges, edges[[1, 0]]], dim=1)

    row, col = edges
    g = g.local_var()
    mask = g.has_edges_between(row, col)

    row_to_remove = row[mask]
    col_to_remove = col[mask]

    eids = g.edge_ids(row_to_remove, col_to_remove)
    g.remove_edges(eids)

    row_to_add = row[~mask]
    col_to_add = col[~mask]

    g.add_edges(row_to_add, col_to_add)

    return g
