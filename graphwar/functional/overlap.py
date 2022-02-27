
import dgl


def graph_overlap(g1: dgl.graph, g2: dgl.graph) -> float:
    """Compute graph overlapping according to
    Node Similarity Preserving Graph Convolutional Networks

    Parameters
    ----------
    g1 : dgl.graph
        a graph
    g2 : dgl.graph
        another graph

    Returns
    -------
    float
        overlapping of the two graphs
    """

    row1, col1 = g1.edges()
    row1 = row1.tolist()
    col1 = col1.tolist()

    row2, col2 = g2.edges()
    row2 = row2.tolist()
    col2 = col2.tolist()

    set_a = set(zip(row1, col1))
    set_b = set(zip(row2, col2))

    return len(set_a-set_b) / len(set_a)
