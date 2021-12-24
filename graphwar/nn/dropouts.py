import torch
import torch.nn as nn
from graphwar.functional import drop_edge, drop_node

class DropEdge(nn.Module):
    """
    DropEdge: Sampling edge using a uniform distribution.
    """    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, g):
        return drop_edge(g, self.p, self.training)
    
class DropNode(nn.Module):
    """
    DropNode: Sampling node using a uniform distribution.
    """
        
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, g):
        return drop_node(g, self.p, self.training)
    
