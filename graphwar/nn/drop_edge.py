import torch
import torch.nn as nn
from graphwar.functional import drop_edge

class DropEdge(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, g):
        return drop_edge(g, self.p, self.training)
