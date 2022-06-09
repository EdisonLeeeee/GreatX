import torch
from torch import nn
from torch import Tensor

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor, Adj
from torch_geometric.nn.inits import zeros

from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor


class MedianConv(nn.Module):
    r"""The graph convolutional operator with median aggregation
    from the `"Understanding Structural Vulnerability 
    in Graph Convolutional Networks"
    <https://www.ijcai.org/proceedings/2021/310>`_ paper (IJCAI'21)

    Parameters
    ----------
    in_channels : int
        dimensions of int samples
    out_channels : int
        dimensions of output samples
    add_self_loops : bool, optional
        whether to add self-loops to the input graph, by default True
    normalize : bool, optional
        whether to compute symmetric normalization
        coefficients on the fly, by default True
    bias : bool, optional
        whether to use bias in the layers, by default True     

    Note
    ----
    The same as :class:`torch_geometric`, our implementation supports:

    * :class:`torch.LongTensor` (recommended): edge indices with shape :obj:`[2, M]`
    * :class:`torch_sparse.SparseTensor`: sparse matrix with sparse shape :obj:`[N, N]`  

    In addition, the arguments :obj:`add_self_loops` and :obj:`normalize` 
    are worked separately. One can set :obj:`normalize=True`  but set
    :obj:`add_self_loops=False`, different from that in :class:`torch_geometric`.                 

    See also
    --------
    :class:`graphwar.nn.models.MedianGCN`       
    """

    def __init__(self, in_channels: int, out_channels: int,
                 add_self_loops: bool = True, normalize: bool = False,
                 bias: bool = True):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        x = self.lin(x)

        # NOTE: we do not support Dense adjacency matrix here
        if isinstance(edge_index, SparseTensor):
            row, col, edge_weight = edge_index.coo()
            edge_index = torch.stack([row, col], dim=0)

        if self.add_self_loops:
            #             edge_index, edge_weight = remove_self_loops(edge_index)
            edge_index, edge_weight = add_self_loops(
                edge_index, num_nodes=x.size(0))

        if self.normalize:
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(0),
                                               improved=False,
                                               add_self_loops=False, dtype=x.dtype)

        out = median_reduce(x, edge_index, edge_weight)

        if self.bias is not None:
            out += self.bias

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


def median_reduce(x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None) -> Tensor:
    # NOTE: `to_dense_batch` requires the `index` is sorted by column
    # TODO: is there any elegant way to avoid `argsort`?
    ix = torch.argsort(edge_index[1])
    edge_index = edge_index[:, ix]
    row, col = edge_index
    x_j = x[row]

    if edge_weight is not None:
        x_j = x_j * edge_weight[ix].unsqueeze(-1)

    dense_x, mask = to_dense_batch(x_j, col)
    h = x_j.new_zeros(dense_x.size(0), dense_x.size(-1))
    deg = mask.sum(dim=1)
    for i in deg.unique():
        deg_mask = deg == i
        h[deg_mask] = dense_x[deg_mask, :i].median(dim=1).values
    return h
