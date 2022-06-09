from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from graphwar.functional import drop_edge, drop_node, drop_path


class DropEdge(nn.Module):
    """
    DropEdge: Sampling edge using a uniform distribution 
    from the `"DropEdge: Towards Deep Graph Convolutional 
    Networks on Node Classification" <https://arxiv.org/abs/1907.10903>`_
    paper (ICLR'20)

    Parameters
    ----------
    p : float, optional
        the probability of dropping out on each edge, by default 0.5

    Returns
    -------
    Tuple[Tensor, Tensor]
        the output edge index and edge weight

    Raises
    ------
    ValueError
        p is out of range [0,1]

    Example
    -------
    >>> from graphwar.nn.layers import DropEdge
    >>> edge_index = torch.LongTensor([[1, 2], [3,4]])
    >>> DropEdge(p=0.5)(edge_index)      

    See also
    --------
    :class:`graphwar.functional.drop_edge`
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        return drop_edge(edge_index, edge_weight, self.p, training=self.training)


class DropNode(nn.Module):
    """
    DropNode: Sampling node using a uniform distribution.
    from the `"Graph Contrastive Learning 
    with Augmentations" <https://arxiv.org/abs/2010.139023>`_
    paper (NeurIPS'20)

    Parameters
    ----------
    p : float, optional
        the probability of dropping out on each node, by default 0.5

    Returns
    -------
    Tuple[Tensor, Tensor]
        the output edge index and edge weight

    Example
    -------
    >>> from graphwar.nn.layers import DropNode
    >>> edge_index = torch.LongTensor([[1, 2], [3,4]])
    >>> DropNode(p=0.5)(edge_index)          

    See also
    --------
    :class:`graphwar.functional.drop_node`    
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        return drop_node(edge_index, edge_weight, self.p, training=self.training)


class DropPath(nn.Module):
    """DropPath: a structured form of :class:`graphwar.functional.drop_edge`.
    From the `"MaskGAE: Masked Graph Modeling Meets 
    Graph Autoencoders" <https://arxiv.org/abs/2205.10053>`_
    paper (arXiv'22)


    Parameters
    ----------
    r : Optional[Union[float, Tensor]], optional
        if :obj:`r` is integer value: the percentage of nodes in the graph that
        chosen as root nodes to perform random walks, by default 0.5
        if :obj:`r` is :class:`torch.Tensor`: a set of custom root nodes
    walks_per_node : int, optional
        number of walks per node, by default 2
    walk_length : int, optional
        number of walk length per node, by default 4
    p : float, optional
        :obj:`p` in random walks, by default 1
    q : float, optional
        :obj:`q` in random walks, by default 1
    num_nodes : int, optional
        number of total nodes in the graph, by default None
    by : str, optional
        sampling root nodes uniformly :obj:`uniform` or 
        by degree distribution :obj:`degree`, by default 'degree'

    Returns
    -------
    Tuple[Tensor, Tensor]
        the output edge index and edge weight

    Raises
    ------
    ImportError
        if :class:`torch_cluster` is not installed.
    ValueError
        :obj:`r` is out of scope [0,1]
    ValueError
        :obj:`r` is not integer value or a Tensor

    Example
    -------
    >>> from graphwar.nn.layers import DropPath
    >>> edge_index = torch.LongTensor([[1, 2], [3,4]])
    >>> DropPath(r=0.5)(edge_index)   

    >>> DropPath(r=torch.tensor([1,2]))(edge_index) # specify root nodes           

    See also
    --------
    :class:`graphwar.functional.drop_path`        
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
