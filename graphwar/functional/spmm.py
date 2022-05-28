import torch
from torch_scatter import scatter

@torch.jit.script
def spmm(x, edge_index, edge_weight=None, reduce='sum'):
    # type: (Tensor, Tensor, Optional[Tensor], str) -> Tensor
    row, col = edge_index[0], edge_index[1]
    x = x if x.dim() > 1 else x.unsqueeze(-1)

    out = x[col]
    if edge_weight is not None:
        out = out * edge_weight.unsqueeze(-1)
    out = scatter(out, row, dim=0, dim_size=x.size(0), reduce=reduce)
    return out