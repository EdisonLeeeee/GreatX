import torch
import torch.nn as nn


class JumpingKnowledge(nn.Module):
    r"""The Jumping Knowledge aggregation module from `Representation Learning on
    Graphs with Jumping Knowledge Networks <https://arxiv.org/abs/1806.03536>`__

    It aggregates the output representations of multiple GNN layers with

    **concatenation**

    .. math::

        h_i^{(1)} \, \Vert \, \ldots \, \Vert \, h_i^{(T)}

    or **max pooling**

    .. math::

        \max \left( h_i^{(1)}, \ldots, h_i^{(T)} \right)

    or **LSTM**

    .. math::

        \sum_{t=1}^T \alpha_i^{(t)} h_i^{(t)}

    with attention scores :math:`\alpha_i^{(t)}` obtained from a BiLSTM

    Parameters
    ----------
    mode : str
        The aggregation to apply. It can be 'cat', 'max', or 'lstm',
        corresponding to the equations above in order.
    in_feats : int, optional
        This argument is only required if :attr:`mode` is ``'lstm'``.
        The output representation size of a single GNN layer. Note that
        all GNN layers need to have the same output representation size.
    num_layers : int, optional
        This argument is only required if :attr:`mode` is ``'lstm'``.
        The number of GNN layers for output aggregation.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> from graphwar.nn import JumpingKnowledge

    >>> # Output representations of two GNN layers
    >>> num_nodes = 3
    >>> in_feats = 4
    >>> feat_list = [torch.zeros(num_nodes, in_feats), torch.ones(num_nodes, in_feats)]

    >>> # Case1
    >>> model = JumpingKnowledge()
    >>> model(feat_list).shape
    torch.Size([3, 8])

    >>> # Case2
    >>> model = JumpingKnowledge(mode='max')
    >>> model(feat_list).shape
    torch.Size([3, 4])

    >>> # Case3
    >>> model = JumpingKnowledge(mode='max', in_feats=in_feats, num_layers=len(feat_list))
    >>> model(feat_list).shape
    torch.Size([3, 4])
    """

    def __init__(self, mode='cat', in_feats=None, num_layers=None):
        super(JumpingKnowledge, self).__init__()
        assert mode in ['cat', 'max', 'lstm'], \
            "Expect mode to be 'cat', or 'max' or 'lstm', got {}".format(mode)
        self.mode = mode

        if mode == 'lstm':
            assert in_feats is not None, 'in_feats is required for lstm mode'
            assert num_layers is not None, 'num_layers is required for lstm mode'
            hidden_size = (num_layers * in_feats) // 2
            self.lstm = nn.LSTM(in_feats, hidden_size,
                                bidirectional=True, batch_first=True)
            self.att = nn.Linear(2 * hidden_size, 1)

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters. This comes into effect only for the lstm mode.
        """
        if self.mode == 'lstm':
            self.lstm.reset_parameters()
            self.att.reset_parameters()

    def forward(self, feat_list):
        r"""

        Description
        -----------
        Aggregate output representations across multiple GNN layers.

        Parameters
        ----------
        feat_list : list[Tensor]
            feat_list[i] is the output representations of a GNN layer.

        Returns
        -------
        Tensor
            The aggregated representations.
        """
        if self.mode == 'cat':
            return torch.cat(feat_list, dim=-1)
        elif self.mode == 'max':
            return torch.stack(feat_list, dim=-1).max(dim=-1)[0]
        else:
            # LSTM
            # (N, num_layers, in_feats)
            stacked_feat_list = torch.stack(feat_list, dim=1)
            alpha, _ = self.lstm(stacked_feat_list)
            alpha = self.att(alpha).squeeze(-1)            # (N, num_layers)
            alpha = torch.softmax(alpha, dim=-1)
            return (stacked_feat_list * alpha.unsqueeze(-1)).sum(dim=1)
