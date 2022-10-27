from typing import Optional, Tuple

import torch
from torch import Tensor

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

from torch_geometric.utils import degree, sort_edge_index, subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes


def drop_edge(edge_index: Tensor, edge_weight: Optional[Tensor] = None,
              p: float = 0.5,
              training: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
    r"""DropEdge: Sampling edge using a uniform distribution
    from the `"DropEdge: Towards Deep Graph Convolutional
    Networks on Node Classification" <https://arxiv.org/abs/1907.10903>`_
    paper (ICLR'20)

    Parameters
    ----------
    edge_index : torch.Tensor
        the input edge index
    edge_weight : Optional[Tensor], optional
        the input edge weight, by default None
    p : float, optional
        the probability of dropping out on each edge, by default 0.5
    training : bool, optional
        whether the model is during training,
        do nothing if :obj:`training=True`, by default True

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

        from greatx.functional import drop_edge
        edge_index = torch.LongTensor([[1, 2], [3,4]])
        drop_edge(edge_index, p=0.5)

    See also
    --------
    :class:`~greatx.nn.layers.DropEdge`
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
        edge_weight = edge_weight[~mask]
    return edge_index, edge_weight


def drop_node(
        edge_index: Tensor, edge_weight: Optional[Tensor] = None,
        p: float = 0.5, training: bool = True,
        num_nodes: Optional[int] = None) -> Tuple[Tensor, Optional[Tensor]]:
    """DropNode: Sampling node using a uniform distribution
    from the `"Graph Contrastive Learning
    with Augmentations" <https://arxiv.org/abs/2010.139023>`_
    paper (NeurIPS'20)

    Parameters
    ----------
    edge_index : torch.Tensor
        the input edge index
    edge_weight : Optional[Tensor], optional
        the input edge weight, by default None
    p : float, optional
        the probability of dropping out on each node, by default 0.5
    training : bool, optional
        whether the model is during training,
        do nothing if :obj:`training=True`, by default True

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

        from greatx.functional import drop_node
        edge_index = torch.LongTensor([[1, 2], [3,4]])
        drop_node(edge_index, p=0.5)

    See also
    --------
    :class:`~greatx.nn.layers.DropNode`
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


def drop_path(edge_index: Tensor, edge_weight: Optional[Tensor] = None,
              p: float = 0.5, walks_per_node: int = 1, walk_length: int = 3,
              num_nodes: Optional[int] = None, start: str = 'node',
              is_sorted: bool = False,
              training: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
    """DropPath: a structured form of :class:`~greatx.functional.drop_edge`
    from the `"MaskGAE: Masked Graph Modeling Meets
    Graph Autoencoders" <https://arxiv.org/abs/2205.10053>`_
    paper (arXiv'22)


    Parameters
    ----------
    edge_index : torch.Tensor
        the input edge index
    edge_weight : Optional[Tensor], optional
        the input edge weight, by default None
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
    training : bool, optional
        whether the model is during training,
        do nothing if :obj:`training=True`, by default True

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

        from greatx.functional import drop_path
        edge_index = torch.LongTensor([[1, 2], [3,4]])
        drop_path(edge_index, p=0.5)

        drop_path(edge_index, p=torch.tensor([1,2])) # specify root nodes


    See also
    --------
    :class:`~greatx.nn.layers.DropPath`
    """

    if torch_cluster is None:
        raise ImportError("`torch_cluster` is not installed.")

    if not training:
        return edge_index, edge_weight
    if p < 0. or p > 1.:
        raise ValueError(f'Sample probability has to be between 0 and 1 '
                         f'(got {p}')

    assert start in ['node', 'edge']
    num_edges = edge_index.size(1)
    edge_mask = edge_index.new_ones(num_edges, dtype=torch.bool)

    if not training or p == 0.0:
        return edge_index, edge_mask

    if random_walk is None:
        raise ImportError('`dropout_path` requires `torch-cluster`.')

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if not is_sorted:
        edge_index = sort_edge_index(edge_index, edge_weight,
                                     num_nodes=num_nodes)
        if edge_weight is not None:
            edge_index, edge_weight = edge_index

    row, col = edge_index
    if start == 'edge':
        sample_mask = torch.rand(row.size(0), device=edge_index.device) <= p
        start = row[sample_mask].repeat(walks_per_node)
    else:
        start = torch.randperm(
            num_nodes,
            device=edge_index.device)[:round(num_nodes *
                                             p)].repeat(walks_per_node)

    deg = degree(row, num_nodes=num_nodes)
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])
    n_id, e_id = random_walk(rowptr, col, start, walk_length, 1.0, 1.0)
    e_id = e_id[e_id != -1].view(-1)  # filter illegal edges
    edge_mask[e_id] = False

    if edge_weight is not None:
        edge_weight = edge_weight[edge_mask]

    return edge_index[:, edge_mask], edge_weight
