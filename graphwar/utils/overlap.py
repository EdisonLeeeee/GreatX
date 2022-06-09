import torch
from torch import Tensor


def overlap(edge_index1: Tensor, edge_index2: Tensor,
            on: str = 'edge', symmetric: bool = False) -> float:
    """Compute graph overlapping according to
    the `"Node Similarity Preserving Graph Convolutional Networks"
    <https://arxiv.org/abs/2011.09643>`_ paper (WSDM'21)

    Parameters
    ----------
    edge_index1 (Tensor) : edges indices of a graph
        a graph
    edge_index2 (Tensor) : edges indices of another graph
        another graph
    on (str) : compute overlap on `edge` or `node`, by default edge

    Returns
    -------
    float
        overlapping of the two graphs on edge or node
    """

    if on == 'edge':
        row1, col1 = edge_index1.tolist()
        row2, col2 = edge_index2.tolist()

        set_a = set(zip(row1, col1))
        set_b = set(zip(row2, col2))

    elif on == 'node':
        set_a = set(edge_index1.flatten().tolist())
        set_b = set(edge_index2.flatten().tolist())

    else:
        raise ValueError(
            f"It currently only supports overlapping on `edge` or `node`, but got {on}.")

    intersection = set_a.intersection(set_b)

    if symmetric:
        return 0.5 * (len(intersection) / len(set_a) + len(intersection) / len(set_b))
    else:
        return len(intersection) / len(set_a)
