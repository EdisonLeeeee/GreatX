import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import DGLError
from dgl.utils import expand_as_pair
from graphwar.utils.transform import dgl_normalize


class RobustConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=False,
                 gamma=1.0,
                 norm='both',
                 add_self_loop=True,
                 activation=None):
        super().__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        self.weight_mean = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.weight_var = nn.Parameter(torch.Tensor(in_feats, out_feats))

        if bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_feats))
            self.bias_var = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_var', None)

        self._gamma = gamma
        self._activation = activation
        self._add_self_loop = add_self_loop
        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        nn.init.xavier_uniform_(self.weight_mean)
        nn.init.xavier_uniform_(self.weight_var)

        if self.bias_mean is not None:
            nn.init.zeros_(self.bias_mean)
            nn.init.zeros_(self.bias_var)

    def forward(self, graph, feat, edge_weight=None):
        feat_mean, feat_var = expand_as_pair(feat)

        mean = feat_mean @ self.weight_mean
        var = feat_var @ self.weight_var

        if self.bias_mean is not None:
            mean = mean + self.bias_mean
            var = var + self.bias_var

        mean = F.relu(mean)
        var = F.relu(var)

        attention = torch.exp(-self._gamma * var)

        if self._add_self_loop:
            graph = graph.add_self_loop()
            if edge_weight is not None:
                size = (graph.num_nodes(),) + edge_weight.size()[1:]
                self_loop = edge_weight.new_ones(size)
                edge_weight = torch.cat([edge_weight, self_loop])
        else:
            graph = graph.local_var()

        norm1 = dgl_normalize(graph, self._norm, edge_weight)
        norm2 = norm1.square()

        graph.ndata['mean'] = mean * attention
        graph.ndata['var'] = var * attention * attention

        graph.edata['_norm1'] = norm1
        graph.edata['_norm2'] = norm2

        graph.update_all(fn.u_mul_e('mean', '_norm1', 'm'), fn.sum('m', 'mean'))
        graph.update_all(fn.u_mul_e('var', '_norm2', 'm'), fn.sum('m', 'var'))

        mean = graph.ndata.pop('mean')
        var = graph.ndata.pop('var')

        if self._activation is not None:
            mean = self._activation(mean)
            var = self._activation(var)

        return mean, var
