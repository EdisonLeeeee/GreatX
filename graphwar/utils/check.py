from typing import Any
import torch


def is_edge_index(x: Any) -> bool:
    """Check if the input :obj:`x` is PyG-like 
    :obj:`edge_index` with shape [2, M], 
    where M is the number of edges.

    Example
    -------
    >>> from graphwar import is_edge_index
    >>> import torch

    >>> edges = torch.LongTensor([[1,2], [3,4]])
    >>> is_edge_index(edges)
    True
    >>> is_edge_index(edges.t()))
    False
    """
    return torch.is_tensor(x) and x.size(0) == 2 and x.dtype == torch.long and x.ndim == 2
