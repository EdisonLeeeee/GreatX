import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.utils import degree, sort_edge_index, to_dense_batch
from torch_scatter import scatter


def spmm(x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None,
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
        * :obj:`sample_median`
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
        out2 = torch.mm(A.t(),x)

        assert torch.allclose(out1, out2)
    """

    if reduce == 'median':
        return scatter_median(x, edge_index, edge_weight)
    elif reduce == 'sample_median':
        return scatter_sample_median(x, edge_index, edge_weight)

    row, col = edge_index
    x = x if x.dim() > 1 else x.unsqueeze(-1)

    out = x[row]
    if edge_weight is not None:
        out = out * edge_weight.unsqueeze(-1)
    out = scatter(out, col, dim=0, dim_size=x.size(0), reduce=reduce)
    return out


def scatter_median(x: Tensor, edge_index: Tensor,
                   edge_weight: OptTensor = None) -> Tensor:
    # NOTE: `to_dense_batch` requires the `index` is sorted by column
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


def scatter_sample_median(x: Tensor, edge_index: Tensor,
                          edge_weight: OptTensor = None) -> Tensor:
    """Approximating the median aggregation with fixed set of
    neighborhood sampling."""

    try:
        from glcore import neighbor_sampler_cpu
    except (ImportError, ModuleNotFoundError):
        raise ModuleNotFoundError(
            "`scatter_sample_median` requires glcore which "
            "is not installed, please refer to "
            "'https://github.com/EdisonLeeeee/glcore' "
            "for more information.")

    if edge_weight is not None:
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  sort_by_row=False)
    else:
        edge_index = sort_edge_index(edge_index, sort_by_row=False)

    row, col = edge_index
    num_nodes = x.size(0)
    deg = degree(col, dtype=torch.long, num_nodes=num_nodes)
    colptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)], dim=0)
    replace = True
    size = int(deg.float().mean().item())
    nodes = torch.arange(num_nodes)
    targets, neighbors, e_id = neighbor_sampler_cpu(colptr.cpu(), row.cpu(),
                                                    nodes, size, replace)

    x_j = x[neighbors]

    if edge_weight is not None:
        x_j = x_j * edge_weight[e_id].unsqueeze(-1)

    return x_j.view(num_nodes, size, -1).median(dim=1).values
