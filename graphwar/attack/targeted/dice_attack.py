import random
from typing import Optional

from graphwar.attack.targeted.random_attack import RandomAttack


class DICEAttack(RandomAttack):
    r"""Implementation of `DICE` attack from the: 
    `"Hiding Individuals and Communities in a Social 
    Network" <https://arxiv.org/abs/1608.00375>`_ paper

    DICE randomly chooses edges to flip based on the principle of
    “Disconnect Internally, Connect Externally” (DICE), 
    which conducts attacks by removing edges between nodes
    with high correlations and connecting edges with low correlations.

    Parameters
    ----------
    data : Data
        PyG-like data denoting the input graph
    device : str, optional
        the device of the attack running on, by default "cpu"
    seed : Optional[int], optional
        the random seed for reproducing the attack, by default None
    name : Optional[str], optional
        name of the attacker, if None, it would be :obj:`__class__.__name__`, 
        by default None
    kwargs : additional arguments of :class:`graphwar.attack.Attacker`,

    Raises
    ------
    TypeError
        unexpected keyword argument in :obj:`kwargs`    

    Example
    -------
    >>> from graphwar.dataset import GraphWarDataset
    >>> import torch_geometric.transforms as T

    >>> dataset = GraphWarDataset(root='~/data/pygdata', name='cora', 
                          transform=T.LargestConnectedComponents())
    >>> data = dataset[0]

    >>> from graphwar.attack.targeted import IGAttack
    >>> attacker = IGAttack(data)
    >>> attacker.reset()
    >>> attacker.attack(target=1) # attacking target node `1` with default budget set as node degree

    >>> attacker.reset()
    >>> attacker.attack(target=1, num_budgets=1) # attacking target node `1` with budget set as 1

    >>> attacker.data() # get attacked graph

    >>> attacker.edge_flips() # get edge flips after attack

    >>> attacker.added_edges() # get added edges after attack

    >>> attacker.removed_edges() # get removed edges after attack    

    Note
    ----
    * Please remember to call :meth:`reset` before each attack.        
    """

    def get_added_edge(self, influence_nodes: list) -> Optional[tuple]:
        u = random.choice(influence_nodes)
        neighbors = self.adjacency_matrix[u].indices.tolist()
        attacker_nodes = list(
            self.nodes_set - set(neighbors) - set([self.target, u]))

        if len(attacker_nodes) == 0:
            return None

        v = random.choice(attacker_nodes)
        label = self.label

        if self.is_legal_edge(u, v) and label[u] != label[v]:
            return (u, v)
        else:
            return None

    def get_removed_edge(self, influence_nodes: list) -> Optional[tuple]:

        u = random.choice(influence_nodes)
        neighbors = self.adjacency_matrix[u].indices.tolist()
        attacker_nodes = list(set(neighbors) - set([self.target, u]))

        if len(attacker_nodes) == 0:
            return None

        v = random.choice(attacker_nodes)

        if self.is_singleton_edge(u, v):
            return None

        label = self.label

        if self.is_legal_edge(u, v) and label[u] == label[v]:
            return (u, v)
        else:
            return None
