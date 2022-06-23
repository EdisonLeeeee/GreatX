from typing import Optional
from copy import copy

import torch
from torch import Tensor
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from graphwar.functional import spmm

class FeaturePropagation(BaseTransform):
    r"""Implementation of FeaturePropagation
    from the `"On the Unreasonable Effectiveness 
    of Feature propagation in Learning 
    on Graphs with Missing Node Features"
    <https://arxiv.org/abs/2111.12128>`_ paper (ICLR'21)


    Parameters
    ----------
    num_iterations : int, optional
        number of iterations to run, by default 40
    missing_mask : Optional[Tensor], optional
        mask on missing features, by default None
    normalize : bool, optional
        whether to compute symmetric normalization
        coefficients on the fly, by default True
    add_self_loops : bool, optional
        whether to add self-loops to the input graph, by default True
        
    Reference
    ---------
    * https://github.com/twitter-research/feature-propagation
        
    """
    def __init__(self, num_iterations: int = 40, 
                 missing_mask: Optional[Tensor] = None,
                 normalize: bool = True,
                 add_self_loops: bool = True):
        super().__init__()
        self.num_iterations = num_iterations
        self.missing_mask = missing_mask
        self.normalize = normalize
        self.add_self_loops = add_self_loops

    def __call__(self, data: Data, inplace: bool = True) -> Data:
        if not inplace:
            data = copy(data)
            
        # out is inizialized to 0 for missing values. However, 
        # its initialization does not matter for the final
        # value at convergence
        out = x = data.x
        known_feature_mask = missing_mask = self.missing_mask
        if missing_mask is not None:
            out = torch.zeros_like(x)
            known_feature_mask = ~missing_mask
            out[known_feature_mask] = x[known_feature_mask]

        edge_index, edge_weight = data.edge_index, data.edge_weight
        
        if self.add_self_loops:
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
            
        if self.normalize:
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(0),
                                            improved=False,
                                            add_self_loops=False,
                                            dtype=x.dtype)
        for _ in range(self.num_iterations):
            # Diffuse current features
            out = spmm(out, edge_index, edge_weight)
            if known_feature_mask is not None:
                # Reset original known features
                out[known_feature_mask] = x[known_feature_mask]
        data.x = out
        
        return data
