import torch
import torch.nn as nn


class DropEdge(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, g):
        if not self.training or not self.p:
            return g
        
        g = g.remove_self_loop()
        num_edges = g.num_edges()
        num_drops = int(self.p * num_edges)
        row, col = g.edges()
        perm = torch.randperm(num_edges, device=g.device)
        g.remove_edges(perm[:num_drops])
        return g.add_self_loop()
            
        
        
        