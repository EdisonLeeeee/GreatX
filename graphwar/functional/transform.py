import math
from typing import Optional, Union

import dgl
import dgl.ops as ops
import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor

__all__ = ['normalize', 'add_self_loop', 'feat_normalize']


def normalize(input: Union[Tensor, dgl.DGLGraph, sp.csr_matrix],
              norm: str = 'both'):
    r"""normalize graph adjacency matrix

    Parameters
    ----------
    input : Union[Tensor, dgl.DGLGraph, sp.csr_matrix]
        the input to normalize, which could be torch.Tensor, 
        dgl.DGLGraph and scipy sparse matrix.
    norm : str, optional
        How to apply the normalizer.  Can be one of the following values:

        * ``both`` (default), where the messages are scaled with :math:`1/c_{ji}`, 
        where :math:`c_{ji}` is the product of the square root of node degrees
        (i.e.,  :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`).

        * ``square``, where the messages are scaled with :math:`1/c_{ji}^2`, where
        :math:`c_{ji}` is defined as above.

        * ``right``, to divide the aggregated messages by each node's in-degrees,
        which is equivalent to averaging the received messages.

        * ``none``, where no normalization is applied.

        * ``left``, to divide the messages sent out from each node by its out-degrees,
        equivalent to random walk normalization.    

    Returns
    -------
    Can be one of the following values:

        * if `input` is torch.Tensor or scipy sparse matrix, then return the normalized adjacency matrix.
        * if `input` is dgl.DGLGraph, then return the normalized edge weight.

    NOTE
    ----
    For dgl.DGLGraph, we only return the normalized edge weight.

    Raises
    ------
    dgl.DGLError
        if `norm` is not one of ('none', 'both', 'square', 'right', 'left')
    TypeError
        if `input` is not one of dgl.DGLGraph, torch.Tensor and scipy sparse_matrix
    """
    if norm not in ('none', 'both', 'right', 'left', 'square'):
        raise dgl.DGLError('Invalid norm value. Must be either "none", "both", "square", "right" or "left".'
                           ' But got "{}".'.format(norm))
    if torch.is_tensor(input):
        return tensor_normalize(input, norm)
    elif isinstance(input, dgl.DGLGraph):
        return dgl_normalize(input, norm)
    elif sp.issparse(input):
        return scipy_normalize(input, norm)
    else:
        raise TypeError("Expected one of dgl.DGLGraph, torch.tensor and scipy.sparse_matrix, "
                        f"but got {type(input)}.")


def feat_normalize(feat: Tensor, norm: str = "standardize",
                   dim: Optional[int] = None,
                   lim_min: float = -1.0, lim_max: float = 1.0):
    """Feature normalization function. Adapted from GRB:
    `https://github.com/THUDM/grb/blob/2f438ccc9e62ffb33a26ca98a95e504985443055/grb/dataset/dataset.py#L638`

    Parameters
    ----------
    feat : Tensor
        node feature matrix with shape [N, D]
    norm : Optional[str], optional
        how to normalize feature matrix, including
        ["linearize", "arctan", "tanh", "standardize", "none"], 
        by default "standardize"
    dim : None or int, optional
        Axis along which the means or standard deviations 
        are computed. The default is to compute the mean or 
        standard deviations of the flattened array, by default None
    lim_min : float, optional
        mininum limit of feature, by default -1.0
    lim_max : float, optional
        maxinum limit of feature, by default 1.0

    Returns
    -------
    feat : Tensor
        normalized feature matrix
    """
    if norm not in ("linearize", "arctan", "tanh", "standardize", "none"):
        raise dgl.DGLError('Invalid norm value. Must be either "linearize", "arctan", "tanh", "standardize" or "none".'
                           ' But got "{}".'.format(norm))

    if norm == 'none':
        return feat

    if norm == "linearize":
        if dim is not None:
            feat_max = feat.max(dim=dim, keepdim=True).values
            feat_min = feat.min(dim=dim, keepdim=True).values
        else:
            feat_max = feat.max()
            feat_min = feat.min()

        k = (lim_max - lim_min) / (feat_max - feat_min)
        feat = lim_min + k * (feat - feat_min)
    else:
        if dim is not None:
            feat_mean = feat.mean(dim=dim, keepdim=True)
            feat_std = feat.std(dim=dim, keepdim=True)
        else:
            feat_mean = feat.mean()
            feat_std = feat.std()

        # standardize
        feat = (feat - feat_mean) / feat_std

        if norm == "arctan":
            feat = 2 * torch.arctan(feat) / math.pi
        elif norm == "tanh":
            feat = torch.tanh(feat)
        elif norm == "standardize":
            pass

    return feat


def add_self_loop(input: Union[Tensor, dgl.DGLGraph, sp.csr_matrix]):
    if torch.is_tensor(input):
        return input + torch.diag(torch.diag(input))
    elif isinstance(input, dgl.DGLGraph):
        return input.add_self_loop()
    elif sp.issparse(input):
        return input + sp.eye(input.shape[0], dtype=input.dtype, format='csr')
    else:
        raise TypeError("Expected one of dgl.DGLGraph, torch.tensor and scipy.sparse_matrix, "
                        f"but got {type(input)}.")


def tensor_normalize(adj_matrix: Tensor, norm: str = 'both'):
    if norm == 'none':
        return adj_matrix

    src_degrees = adj_matrix.sum(dim=0).clamp(min=1)
    dst_degrees = adj_matrix.sum(dim=1).clamp(min=1)

    if norm == 'left':
        # A * D^-1
        norm_src = (1.0 / src_degrees).view(1, -1)
        adj_matrix = adj_matrix * norm_src
    elif norm == 'right':
        # D^-1 * A
        norm_dst = (1.0 / dst_degrees).view(-1, 1)
        adj_matrix = adj_matrix * norm_dst
    else:  # both or square
        if norm == 'both':
            # D^-0.5 * A * D^-0.5
            pow = -0.5
        else:
            # D^-1 * A * D^-1
            pow = -1
        norm_src = torch.pow(src_degrees, pow).view(1, -1)
        norm_dst = torch.pow(dst_degrees, pow).view(-1, 1)
        adj_matrix = norm_src * adj_matrix * norm_dst
    return adj_matrix


def dgl_normalize(g: dgl.DGLGraph, norm: str = 'both', edge_weight=None):
    e_norm = torch.ones(
        g.num_edges(), device=g.device) if edge_weight is None else edge_weight

    if norm == 'none':
        return e_norm

    if edge_weight is None:
        src_degrees = g.in_degrees().clamp(min=1)
        dst_degrees = g.out_degrees().clamp(min=1)
    else:
        # a weighted graph
        src_degrees = ops.copy_e_sum(g, edge_weight)
        dst_degrees = ops.copy_e_sum(dgl.reverse(g), edge_weight)
    if norm == 'left':
        # A * D^-1
        norm_src = 1.0 / src_degrees
        e_norm = ops.e_mul_v(g, e_norm, norm_src)
    elif norm == 'right':
        # D^-1 * A
        norm_dst = 1.0 / dst_degrees
        e_norm = ops.e_mul_u(g, e_norm, norm_dst)
    else:  # both or square
        if norm == 'both':
            # D^-0.5 * A * D^-0.5
            pow = -0.5
        else:
            # D^-1 * A * D^-1
            pow = -1
        norm_src = torch.pow(src_degrees, pow)
        norm_dst = torch.pow(dst_degrees, pow)
        e_norm = ops.e_mul_u(g, e_norm, norm_src)
        e_norm = ops.e_mul_v(g, e_norm, norm_dst)
    return e_norm


def scipy_normalize(adj_matrix: sp.csr_matrix, norm: str = 'both'):
    if norm == 'none':
        return adj_matrix

    src_degrees = np.maximum(adj_matrix.sum(0).A1, 1)
    dst_degrees = np.maximum(adj_matrix.sum(1).A1, 1)

    if norm == 'left':
        # A * D^-1
        norm_src = sp.diags(1.0 / src_degrees)
        adj_matrix = adj_matrix @ norm_src
    elif norm == 'right':
        # D^-1 * A
        norm_dst = sp.diags(1.0 / dst_degrees)
        adj_matrix = norm_dst @ adj_matrix
    else:  # both or square
        if norm == 'both':
            # D^-0.5 * A * D^-0.5
            pow = -0.5
        else:
            # D^-1 * A * D^-1
            pow = -1
        norm_src = sp.diags(np.power(src_degrees, pow))
        norm_dst = sp.diags(np.power(dst_degrees, pow))
        adj_matrix = norm_src @ adj_matrix @ norm_dst
    return adj_matrix
