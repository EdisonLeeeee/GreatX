from typing import Optional, Union

import torch
from torch import Tensor


def normalize(feat: Tensor, norm: str = "standardize",
              dim: Optional[int] = None,
              lim_min: float = -1.0, lim_max: float = 1.0) -> Tensor:
    """Feature normalization function. 

    Adapted from GRB:
    https://github.com/THUDM/grb/blob/master/grb/dataset/dataset.py#L638

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
        minimum limit of feature, by default -1.0
    lim_max : float, optional
        maximum limit of feature, by default 1.0

    Returns
    -------
    feat : Tensor
        normalized feature matrix
    """
    if norm not in ("linearize", "arctan", "tanh", "standardize", "none"):
        raise ValueError('Invalid norm value. Must be either "linearize", "arctan", "tanh", "standardize" or "none".'
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
