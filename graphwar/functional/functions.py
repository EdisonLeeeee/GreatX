from typing import Optional

import torch

from .coalesce import coalesce


def pairwise_cosine_similarity(X: torch.Tensor,
                               Y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute cosine similarity between samples in X and Y.

    Cosine similarity, or the cosine kernel, computes similarity as the
    normalized dot product of X and Y:

        K(X, Y) = <X, Y> / (||X||*||Y||)

    On L2-normalized data, this function is equivalent to linear_kernel.


    Parameters
    ----------
    X : torch.Tensor, shape (N, M)
        Input data.
    Y : Optional[torch.Tensor], optional
        Input data. If ``None``, the output will be the pairwise
        similarities between all samples in ``X``., by default None

    Returns
    -------
    torch.Tensor, shape (N, M)
        the pairwise similarities matrix
    """

    A_norm = X / X.norm(dim=1)[:, None]
    if Y is None:
        B_norm = A_norm
    else:
        B_norm = Y / Y.norm(dim=1)[:, None]
    S = torch.mm(A_norm, B_norm.transpose(0, 1))
    return S


def attr_sim(x, k=5):
    x = x.bool().float()  # x[x!=0] = 1

    sims = pairwise_cosine_similarity(x)
    indices_sorted = sims.argsort(1)
    selected = torch.cat((indices_sorted[:, :k],
                          indices_sorted[:, - k - 1:]), dim=1)
    row = torch.arange(x.size(0), device=x.device).repeat_interleave(
        selected.size(1))
    col = selected.view(-1)

    mask = row != col
    row, col = row[mask], col[mask]

    mask = row > col
    row[mask], col[mask] = col[mask].clone(), row[mask].clone()

    node_pairs = torch.stack([row, col], dim=0)
    node_pairs = coalesce(node_pairs, num_nodes=x.size(0))
    return sims[node_pairs[0], node_pairs[1]], node_pairs
