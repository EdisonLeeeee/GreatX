import random
from typing import Optional

import dgl

from graphwar.attack.untargeted.random_attack import RandomAttack


class DICEAttack(RandomAttack):
    """DICE attacker that randomly chooses edges to flip 
    based on “Disconnect Internally, Connect Externally” (DICE), 
    which conducts attacks by removing edges between nodes
    with high correlations and connecting edges with low correlations.

    Reference
    --------
    [1] M. Waniek, T. P. Michalak, M. J. Wooldridge, and T. Rahwan, 
    “Hiding individuals and communities in a social network,” 
    Nature Human Behaviour, vol. 2, no. 2, pp. 139–147, 2018.
    """

    def __init__(self, graph: dgl.DGLGraph,
                 device: str = "cpu", seed: Optional[int] = None, name: Optional[str] = None, **kwargs):
        super().__init__(graph=graph, device=device, seed=seed, name=name, **kwargs)
        self._check_node_label_exists()

    def get_added_edge(self, influence_nodes: list) -> Optional[tuple]:
        u = random.choice(influence_nodes)
        neighbors = self.adjacency_matrix[u].indices.tolist()
        attacker_nodes = list(self.nodes_set - set(neighbors + [u]))

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
        # assume that the graph has no self-loops
        attacker_nodes = list(set(neighbors))

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
