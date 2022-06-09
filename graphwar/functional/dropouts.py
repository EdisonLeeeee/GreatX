from typing import Optional, Tuple, Union

import torch
from torch import Tensor
try:
    import torch_cluster
except ImportError:
    torch_cluster = None

from torch_geometric.utils import subgraph, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes


def drop_edge(edge_index: Tensor, edge_weight: Optional[Tensor] = None,
              p: float = 0.5, training: bool = True) -> Tuple[Tensor, Tensor]:
    """
    DropEdge: Sampling edge using a uniform distribution 
    from the `"DropEdge: Towards Deep Graph Convolutional 
    Networks on Node Classification" <https://arxiv.org/abs/1907.10903>`_
    paper (ICLR'20)

    Parameters
    ----------
    edge_index : Tensor
        the input edge index
    edge_weight : Optional[Tensor], optional
        the input edge weight, by default None
    p : float, optional
        the probability of dropping out on each edge, by default 0.5
    training : bool, optional
        whether the model is during training,
        do nothing if :obj:`training=True`,, by default True

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
    >>> from graphwar.functional import drop_edge
    >>> edge_index = torch.LongTensor([[1, 2], [3,4]])
    >>> drop_edge(edge_index, p=0.5)      

    See also
    --------
    :class:`graphwar.nn.layers.DropEdge`         
    """

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or not p:
        return edge_index, edge_weight

    num_edges = edge_index.size(1)
    e_ids = torch.arange(num_edges, dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    edge_index = edge_index[:, ~mask]
    if edge_weight is not None:
        edge_weight = edge_weight[:, ~mask]
    return edge_index, edge_weight


def drop_node(edge_index: Tensor, edge_weight: Optional[Tensor] = None,
              p: float = 0.5, training: bool = True,
              num_nodes: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """
    DropNode: Sampling node using a uniform distribution.
    from the `"Graph Contrastive Learning 
    with Augmentations" <https://arxiv.org/abs/2010.139023>`_
    paper (NeurIPS'20)

    Parameters
    ----------
    edge_index : Tensor
        the input edge index
    edge_weight : Optional[Tensor], optional
        the input edge weight, by default None
    p : float, optional
        the probability of dropping out on each node, by default 0.5
    training : bool, optional
        whether the model is during training,
        do nothing if :obj:`training=True`,, by default True

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
    >>> from graphwar.functional import drop_node
    >>> edge_index = torch.LongTensor([[1, 2], [3,4]])
    >>> drop_node(edge_index, p=0.5)

    See also
    --------
    :class:`graphwar.nn.layers.DropNode`   
    """

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or not p:
        return edge_index, edge_weight

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    nodes = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(nodes, 1 - p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    subset = nodes[mask]
    return subgraph(subset, edge_index, edge_weight)


def drop_path(edge_index: Tensor,
              edge_weight: Optional[Tensor] = None,
              r: Optional[Union[float, Tensor]] = 0.5,
              walks_per_node: int = 2,
              walk_length: int = 4,
              p: float = 1, q: float = 1,
              training: bool = True,
              num_nodes: int = None,
              by: str = 'degree') -> Tuple[Tensor, Tensor]:
    """DropPath: a structured form of :class:`graphwar.functional.drop_edge`.
    From the `"MaskGAE: Masked Graph Modeling Meets 
    Graph Autoencoders" <https://arxiv.org/abs/2205.10053>`_
    paper (arXiv'22)


    Parameters
    ----------
    edge_index : Tensor
        the input edge index
    edge_weight : Optional[Tensor], optional
        the input edge weight, by default None
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
    training : bool, optional
        whether the model is during training,
        do nothing if :obj:`training=True`, by default True
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
    >>> from graphwar.functional import drop_path
    >>> edge_index = torch.LongTensor([[1, 2], [3,4]])
    >>> drop_path(edge_index, r=0.5)   

    >>> drop_path(edge_index, r=torch.tensor([1,2])) # specify root nodes   


    See also
    --------
    :class:`graphwar.nn.layers.DropPath`      
    """

    if torch_cluster is None:
        raise ImportError("`torch_cluster` is not installed.")

    if not training:
        return edge_index, edge_weight

    assert by in {'degree', 'uniform'}
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float)

    if isinstance(r, (int, float)):
        if r < 0. or r > 1.:
            raise ValueError(f'Root node sampling ratio `r` has to be between 0 and 1 '
                             f'(got {r}')
        if r == 0.:
            return edge_index, edge_weight

        num_starts = int(r * num_nodes)
        if by == 'degree':
            prob = deg / deg.sum()
            start = prob.multinomial(num_samples=num_starts, replacement=True)
        else:
            start = torch.randperm(num_nodes, device=edge_index.device)[
                :num_starts]
    elif torch.is_tensor(r):
        start = r.to(edge_index)
    else:
        raise ValueError('Root node sampling ratio `r` must be '
                         f'`float`, `torch.Tensor`, but got {r}.')

    if walks_per_node:
        start = start.repeat(walks_per_node)

    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])

    n_id, e_id = torch.ops.torch_cluster.random_walk(
        rowptr, col, start, walk_length, p, q)
    mask = row.new_ones(row.size(0), dtype=torch.bool)
    mask[e_id.view(-1)] = False

    if edge_weight is not None:
        edge_weight = edge_weight[mask]
    return edge_index[:, mask], edge_weight
