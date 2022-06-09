import random
from typing import Optional

from tqdm import tqdm

from graphwar.attack.targeted.targeted_attacker import TargetedAttacker


class RandomAttack(TargetedAttacker):
    r"""Random attacker that randomly chooses edges to flip.

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

    >>> from graphwar.attack.targeted import RandomAttack
    >>> attacker = RandomAttack(data)
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

    def attack(self,
               target, *,
               num_budgets=None,
               threshold=0.5,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(target, target_label=None, num_budgets=num_budgets,
                       direct_attack=direct_attack, structure_attack=structure_attack,
                       feature_attack=feature_attack)
        assert 0 < threshold < 1, f"'threshold' should to be greater than 0 and less than 1, but got {threshold}."

        if direct_attack:
            influence_nodes = [target]
        else:
            influence_nodes = self.adjacency_matrix[target].indices.tolist()

        num_chosen = 0

        with tqdm(total=self.num_budgets, desc='Peturbing graph...', disable=disable) as pbar:
            while num_chosen < self.num_budgets:
                # randomly choose to add or remove edges
                if random.random() <= threshold:
                    delta = 1
                    edge = self.get_added_edge(influence_nodes)
                else:
                    delta = -1
                    edge = self.get_removed_edge(influence_nodes)

                if edge is not None:
                    u, v = edge
                    if delta > 0:
                        self.add_edge(u, v, num_chosen)
                    else:
                        self.remove_edge(u, v, num_chosen)

                    num_chosen += 1
                    pbar.update(1)

        return self

    def get_added_edge(self, influence_nodes: list) -> Optional[tuple]:
        u = random.choice(influence_nodes)
        neighbors = self.adjacency_matrix[u].indices.tolist()
        attacker_nodes = list(
            self.nodes_set - set(neighbors) - set([self.target, u]))

        if len(attacker_nodes) == 0:
            return None

        v = random.choice(attacker_nodes)

        if self.is_legal_edge(u, v):
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

        if self.is_legal_edge(u, v):
            return (u, v)
        else:
            return None
