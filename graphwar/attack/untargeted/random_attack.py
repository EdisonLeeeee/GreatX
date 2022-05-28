from typing import Optional

import random
import numpy as np
from tqdm import tqdm

from graphwar.attack.untargeted.untargeted_attacker import UntargetedAttacker

class RandomAttack(UntargetedAttacker):
    """Random attacker that randomly chooses edges to flip."""

    def attack(self,
               num_budgets=0.05, *,
               threshold=0.5,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(num_budgets=num_budgets,
                       structure_attack=structure_attack,
                       feature_attack=feature_attack)
        assert 0 < threshold < 1, f"'threshold' should to be greater than 0 and less than 1, but got {threshold}."
        random_arr = np.random.choice(2, self.num_budgets,
                                      p=[1 - threshold, threshold]) * 2 - 1

        influence_nodes = list(self.nodes_set)
        for it, remove_or_insert in tqdm(enumerate(random_arr),
                                         desc='Peturbing Graph',
                                         disable=disable):
            # randomly choose to add or remove edges
            if remove_or_insert > 0:
                edge = self.get_added_edge(influence_nodes)
                while edge is None:
                    edge = self.get_added_edge(influence_nodes)
                u, v = edge
                self.add_edge(u, v, it)

            else:
                edge = self.get_removed_edge(influence_nodes)
                while edge is None:
                    edge = self.get_removed_edge(influence_nodes)
                u, v = edge
                self.remove_edge(u, v, it)

        return self

    def get_added_edge(self, influence_nodes: list) -> Optional[tuple]:
        u = random.choice(influence_nodes)
        neighbors = self.adjacency_matrix[u].indices.tolist()
        attacker_nodes = list(self.nodes_set - set(neighbors + [u]))

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
        # assume that the graph has no self-loops
        attacker_nodes = list(set(neighbors))

        if len(attacker_nodes) == 0:
            return None

        v = random.choice(attacker_nodes)

        if self.is_singleton_edge(u, v):
            return None

        if self.is_legal_edge(u, v):
            return (u, v)
        else:
            return None
