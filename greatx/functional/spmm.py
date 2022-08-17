import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.typing import OptTensor, Adj
from torch_geometric.utils import to_dense_batch


def spmm(x: Tensor, edge_index: Tensor,
         edge_weight: OptTensor = None,
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
        reduction of the sparse matrix multiplication, including:
        * :obj:`mean`
        * :obj:`sum`
        * :obj:`max`
        * :obj:`min`
        * :obj:`median`
        by default :obj:'sum'

    Returns
    -------
    Tensor
        the output result of the multiplication.

    Example
    -------
    .. code-block:: python

        import torch
        from greatx.functional import spmm

        x = torch.randn(5,2)
        edge_index = torch.LongTensor([[1,2], [3,4]])
        out1 = spmm(x, edge_index, reduce='sum')

        # which is equivalent to:
        A = torch.zeros(5,5)
        A[edge_index[0], edge_index[1]] = 1.0
        out2 = torch.mm(A,t(),x)

        assert torch.allclose(out1, out2)
    """

    if reduce == 'median':
        return scatter_median(x, edge_index, edge_weight)

    row, col = edge_index[0], edge_index[1]
    x = x if x.dim() > 1 else x.unsqueeze(-1)

    out = x[row]
    if edge_weight is not None:
        out = out * edge_weight.unsqueeze(-1)
    out = scatter(out, col, dim=0, dim_size=x.size(0), reduce=reduce)
    return out


def scatter_median(x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None) -> Tensor:
    # NOTE: `to_dense_batch` requires the `index` is sorted by column
    # TODO: is there any elegant way to avoid `argsort`?
    ix = torch.argsort(edge_index[1])
    edge_index = edge_index[:, ix]
    row, col = edge_index
    x_j = x[row]

    if edge_weight is not None:
        x_j = x_j * edge_weight[ix].unsqueeze(-1)

    dense_x, mask = to_dense_batch(x_j, col, batch_size=x.size(0))
    h = x_j.new_zeros(dense_x.size(0), dense_x.size(-1))
    deg = mask.sum(dim=1)
    for i in deg.unique():
        if i == 0:
            continue
        deg_mask = deg == i
        h[deg_mask] = dense_x[deg_mask, :i].median(dim=1).values
    return h
