import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from graphwar.nn import Linear


class RobustConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=False,
                 gamma=1.0,
                 activation=None):
        super().__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.linear_mean = Linear(in_feats, out_feats, bias=bias)
        self.linear_var = Linear(in_feats, out_feats, bias=bias)

        self._gamma = gamma
        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""

        self.linear_mean.reset_parameters()
        self.linear_var.reset_parameters()

    def forward(self, graph, feat):
        if not isinstance(feat, tuple):
            feat = (feat, feat)

        mean = self.linear_mean(feat[0])
        var = self.linear_var(feat[1])

        mean = F.relu(mean)
        var = F.relu(var)

        attention = torch.exp(-self._gamma * var)

        degs = graph.in_degrees().float().clamp(min=1)
        norm1 = torch.pow(degs, -0.5).to(mean.device).unsqueeze(1)
        norm2 = norm1.square()

        with graph.local_scope():
            graph.ndata['mean'] = mean * attention * norm1
            graph.ndata['var'] = var * attention * attention * norm2
            graph.update_all(fn.copy_src('mean', 'm'), fn.sum('m', 'mean'))
            graph.update_all(fn.copy_src('var', 'm'), fn.sum('m', 'var'))

            mean = graph.ndata.pop('mean') * norm1
            var = graph.ndata.pop('var') * norm2

            if self._activation is not None:
                mean = self._activation(mean)
                var = self._activation(var)

        return mean, var
