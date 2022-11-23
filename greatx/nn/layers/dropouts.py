from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from greatx.functional import drop_edge, drop_node, drop_path


class DropEdge(nn.Module):
    """DropEdge: Sampling edge using a uniform distribution
    from the `"DropEdge: Towards Deep Graph Convolutional
    Networks on Node Classification" <https://arxiv.org/abs/1907.10903>`_
    paper (ICLR'20)

    Parameters
    ----------
    p : float, optional
        the probability of dropping out on each edge, by default 0.5

    Returns
    -------
    Tuple[Tensor, Optional[Tensor]]
        the output edge index and edge weight

    Raises
    ------
    ValueError
        p is out of range [0,1]

    Example
    -------
    .. code-block:: python

        from greatx.nn.layers import DropEdge
        edge_index = torch.LongTensor([[1, 2], [3,4]])
        DropEdge(p=0.5)(edge_index)

    See also
    --------
    :class:`greatx.functional.drop_edge`
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(
        self,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """"""
        return drop_edge(edge_index, edge_weight, self.p,
                         training=self.training)


class DropNode(nn.Module):
    """DropNode: Sampling node using a uniform distribution
    from the `"Graph Contrastive Learning
    with Augmentations" <https://arxiv.org/abs/2010.139023>`_
    paper (NeurIPS'20)

    Parameters
    ----------
    p : float, optional
        the probability of dropping out on each node, by default 0.5

    Returns
    -------
    Tuple[Tensor, Optional[Tensor]]
        the output edge index and edge weight

    Example
    -------
    .. code-block:: python

        from greatx.nn.layers import DropNode
        edge_index = torch.LongTensor([[1, 2], [3,4]])
        DropNode(p=0.5)(edge_index)

    See also
    --------
    :class:`greatx.functional.drop_node`
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(
        self,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """"""
        return drop_node(edge_index, edge_weight, self.p,
                         training=self.training)


class DropPath(nn.Module):
    """DropPath: a structured form of :class:`greatx.functional.drop_edge`
    from the `"MaskGAE: Masked Graph Modeling Meets
    Graph Autoencoders" <https://arxiv.org/abs/2205.10053>`_
    paper (arXiv'22)


    Parameters
    ----------
    p : Optional[Union[float, Tensor]], optional
        If :obj:`p` is a float value - the percentage of
        nodes in the graph that chosen as root nodes to
        perform random walks.
        If :obj:`p` is :class:`torch.Tensor` - a set of
        custom root nodes.
        By default, :obj:`p=0.5`.
    walks_per_node : int, optional
        number of walks per node, by default 1
    walk_length : int, optional
        number of walk length per node, by default 3
    num_nodes : int, optional
        number of total nodes in the graph, by default None
    start : string, optional
        the type of starting node chosen from node of edge,
        by default 'node'
    is_sorted : bool, optional
        whether the input :obj:`edge_index` is sorted

    Returns
    -------
    Tuple[Tensor, Optional[Tensor]]
        the output edge index and edge weight

    Raises
    ------
    ImportError
        if :class:`torch_cluster` is not installed.
    ValueError
        :obj:`p` is out of scope [0,1]
    ValueError
        :obj:`p` is not integer value or a Tensor

    Example
    -------
    .. code-block:: python

        from greatx.nn.layers import DropPath
        edge_index = torch.LongTensor([[1, 2], [3,4]])
        DropPath(p=0.5)(edge_index)

        DropPath(p=torch.tensor([1,2]))(edge_index) # specify root nodes

    See also
    --------
    :class:`greatx.functional.drop_path`
    """
    def __init__(self, p: float = 0.5, walks_per_node: int = 1,
                 walk_length: int = 3, num_nodes: Optional[int] = None,
                 start: str = 'node', is_sorted: bool = False):
        super().__init__()
        self.p = p
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.num_nodes = num_nodes
        self.start = start
        self.is_sorted = is_sorted

    def forward(
        self,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """"""
        return drop_path(edge_index, edge_weight, p=self.p,
                         walks_per_node=self.walks_per_node,
                         walk_length=self.walk_length,
                         num_nodes=self.num_nodes, start=self.start,
                         training=self.training)
