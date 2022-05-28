import torch

def is_edge_index(x):
    """Check if the input `x` is edge_index with shape [2, M]."""
    return torch.is_tensor(x) and x.size(0) == 2 and x.dtype == torch.long and x.ndim == 2 