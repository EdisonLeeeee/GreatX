from graphwar.models.gcn import GCN
import torch
import torch.nn as nn
import dgl.function as fn

from graphwar.config import Config
from graphwar.nn import GCNConv, Sequential, activations
from graphwar.utils import wrapper

_EDGE_WEIGHT = Config.edge_weight


class JKNet(nn.Module):
    """Graph Convolution Network with Jumping knowledge (JKNet)

    Example
    -------
    # JKNet with five hidden layers
    >>> model = JKNet(100, 10, hids=[16]*5)
    """

    @wrapper
    def __init__(self,
                 in_feats: int,
                 out_feats: int,
                 hids: list = [16]*3,
                 acts: list = ['relu']*3,
                 dropout: float = 0.5,
                 mode: str = 'cat',
                 bn: bool = False,
                 bias: bool = True,
                 norm: str = 'both'):
        r"""
        Parameters
        ----------
        in_feats : int, 
            the input dimmensions of model
        out_feats : int, 
            the output dimensions of model
        hids : list, optional
            the number of hidden units of each hidden layer, by default [16, 16, 16]
        acts : list, optional
            the activaction function of each hidden layer, by default ['relu', 'relu', 'relu']
        dropout : float, optional
            the dropout ratio of model, by default 0.5
        mode : str, optional
            the mode of jumping knowledge, including 'cat', 'lstm', 'max' and 'last',
            by default 'cat'
        bias : bool, optional
            whether to use bias in the layers, by default True
        bn: bool, optional
            whether to use `BatchNorm1d` after the convolution layer, by default False
        norm : str, optional
            How to apply the normalizer.  Can be one of the following values:

            * ``both``, where the messages are scaled with :math:`1/c_{ji}`, 
            where :math:`c_{ji}` is the product of the square root of node degrees
            (i.e.,  :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`).

            * ``square``, where the messages are scaled with :math:`1/c_{ji}^2`, where
            :math:`c_{ji}` is defined as above.

            * ``right``, to divide the aggregated messages by each node's in-degrees,
            which is equivalent to averaging the received messages.

            * ``none``, where no normalization is applied.

            * ``left``, to divide the messages sent out from each node by its out-degrees,
            equivalent to random walk normalization.            
        """

        super().__init__()
        assert mode in {'cat', 'lstm', 'max', 'last'}
        self.mode = mode
        num_JK_layers = len(list(hids)) - 1  # number of JK layers

        assert num_JK_layers >= 1 and len(set(
            hids)) == 1, 'the number of hidden layers should be greater than 2 and the hidden units must be equal'

        conv = []
        assert len(hids) == len(acts)
        for hid, act in zip(hids, acts):
            block = []
            block.append(nn.Dropout(dropout))
            block.append(GCNConv(in_feats,
                                 hid,
                                 bias=bias, norm=norm,
                                 activation=None))
            if bn:
                block.append(nn.BatchNorm1d(hid))
            block.append(activations.get(act))
            conv.append(Sequential(*block, loc=1))
            in_feats = hid

        # `loc=1` specifies the location of features.
        self.conv = Sequential(*conv, loc=1)

        assert len(conv) == num_JK_layers + 1

        if self.mode == 'cat':
            hid = hid * (num_JK_layers + 1)
        elif self.mode == 'lstm':
            self.lstm = nn.LSTM(hid, (num_JK_layers * hid) //
                                2, bidirectional=True, batch_first=True)
            self.attn = nn.Linear(2 * ((num_JK_layers * hid) // 2), 1)

        self.mlp = nn.Linear(hid, out_feats)

    def reset_parameters(self):
        self.conv.reset_parameters()
        if self.mode == 'lstm':
            self.lstm.reset_parameters()
            self.attn.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, g, feat, edge_weight=None):

        if edge_weight is None:
            edge_weight = g.edata.get(_EDGE_WEIGHT, edge_weight)

        feat_list = []
        for conv in self.conv:
            feat = conv(g, feat, edge_weight=edge_weight)
            feat_list.append(feat)

        if self.mode == 'cat':
            out = torch.cat(feat_list, dim=-1)
        elif self.mode == 'max':
            out = torch.stack(feat_list, dim=-1).max(dim=-1)[0]
        elif self.mode == 'last':
            out = feat_list[-1]
        else:
            # lstm
            x = torch.stack(feat_list, dim=1)
            alpha, _ = self.lstm(x)
            alpha = self.attn(alpha).squeeze(-1)
            alpha = torch.softmax(alpha, dim=-1).unsqueeze(-1)
            out = (x * alpha).sum(dim=1)

        g = g.local_var()
        g.ndata['h'] = out
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))

        return self.mlp(g.ndata['h'])
