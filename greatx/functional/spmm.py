from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter


# @torch.jit.script
def spmm(x: Tensor, edge_index: Tensor,
         edge_weight: Optional[Tensor] = None,
         reduce: str = 'sum') -> Tensor:
    r"""Sparse matrix multiplication using :class:`torch_scatter`.

    Parameters
    ----------
    x : torch.Tensor
        the input dense 2D-matrix
    edge_index : torch.Tensor
        the location of the non-zeros elements in the sparse matrix,
        denoted as :obj:`edge_index` with shape [2, M]
    edge_weight : Optional[Tensor], optional
        the edge weight of the sparse matrix, by default None
    reduce : str, optional
        reduction of the sparse matrix multiplication, by default 'sum'

    Returns
    -------
    Tensor
        the output result of the multiplication.

    Example
    -------
    >>> import torch
    >>> from greatx.functional import spmm

    >>> x = torch.randn(5,2)
    >>> edge_index = torch.LongTensor([[1,2], [3,4]])
    >>> spmm(x, edge_index, reduce='sum')

    >>> # which is equivalent to:
    >>> A = torch.zeros(5,5)
    >>> A[edge_index[0], edge_index[1]] = 1.0
    >>> torch.mm(A,t(),x)
    """
    row, col = edge_index[0], edge_index[1]
    x = x if x.dim() > 1 else x.unsqueeze(-1)

    out = x[row]
    if edge_weight is not None:
        out = out * edge_weight.unsqueeze(-1)
    out = scatter(out, col, dim=0, dim_size=x.size(0), reduce=reduce)
    return out
