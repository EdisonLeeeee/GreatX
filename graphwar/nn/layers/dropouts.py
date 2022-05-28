from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from graphwar.functional import drop_edge, drop_node, drop_path


class DropEdge(nn.Module):
    """
    DropEdge: Sampling edge using a uniform distribution.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        return drop_edge(edge_index, edge_weight, self.p, training=self.training)


class DropNode(nn.Module):
    """
    DropNode: Sampling node using a uniform distribution.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        return drop_node(edge_index, edge_weight, self.p, training=self.training)


class DropPath(nn.Module):
    """
    DropPath: path-wise structured dropout.
    """

    def __init__(self, r: float = 0.5,
                 walks_per_node: int = 2,
                 walk_length: int = 4,
                 p: float = 1, q: float = 1,
                 num_nodes: int = None,
                 by: str = 'degree'):
        super().__init__()
        self.r = r
        self.walks_per_node = walks_per_node
        self, walk_length = walk_length
        self.p = p
        self.q = q
        self.num_nodes = num_nodes
        self.by = by

    def forward(self, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        return drop_path(edge_index, edge_weight, r=self.r, p=self.p, q=self.q,
                         num_nodes=self.num_nodes, by=self.by, training=self.training)
