import dgl
import torch
import warnings
import numpy as np
from torch import Tensor

from typing import Optional, Union
from functools import lru_cache

from graphwar.utils import BunchDict
from graphwar.attack.attacker import Attacker


class FlipAttacker(Attacker):
    """Adversarial attacker for graph data by edge flipping.
    """
    _max_perturbations: Union[float, int] = 0
    _allow_feature_attack: bool = False
    _allow_structure_attack: bool = True
    _allow_singleton: bool = False

    def reset(self) -> "FlipAttacker":
        super().reset()
        self.g.cache_clear()
        self._removed_edges = {}
        self._added_edges = {}
        self._removed_feats = {}
        self._added_feats = {}
        self.degree = self._degree.clone().to(self.device)
        return self

    def remove_edge(self, u: int, v: int, it: Optional[int] = None):
        """remove one edge from the graph.

        Parameters
        ----------
        u : int
            the src node of the edge
        v : int
            the dst node of the edge
        it : Optional[int], optional
            the iteration that indicates the order of the edge being removed, by default None
        """
        if not self._allow_singleton:
            is_singleton_u = self.degree[u] <= 1
            is_singleton_v = self.degree[v] <= 1

            if is_singleton_u or is_singleton_v:
                warnings.warn(f"You are trying to remove an edge ({u}-{v}) that would result in singleton nodes. If the behavior is not intended, please make sure you have set `attacker.set_allow_singleton(False)` or check your algorithm.", UserWarning)

        self._removed_edges[(u, v)] = it
        self.degree[u] -= 1
        self.degree[v] -= 1

    def add_edge(self, u: int, v: int, it: Optional[int] = None):
        """add one edge to the graph.

        Parameters
        ----------
        u : int
            the src node of the edge
        v : int
            the dst node of the edge
        it : Optional[int], optional
            the iteration that indicates the order of the edge being added, by default None
        """
        self._added_edges[(u, v)] = it
        self.degree[u] += 1
        self.degree[v] += 1

    def removed_edges(self) -> Optional[Tensor]:
        edges = self._removed_edges
        if edges is None or len(edges) == 0:
            return None

        if torch.is_tensor(edges):
            return edges.to(self.device)

        if isinstance(edges, dict):
            edges = list(edges.keys())

        return torch.tensor(np.asarray(edges, dtype="int64").T, device=self.device)

    def added_edges(self) -> Optional[Tensor]:
        edges = self._added_edges
        if edges is None or len(edges) == 0:
            return None

        if torch.is_tensor(edges):
            return edges.to(self.device)

        if isinstance(edges, dict):
            edges = list(edges.keys())

        return torch.tensor(np.asarray(edges, dtype="int64").T, device=self.device)

    def edge_flips(self) -> BunchDict:
        added = self.added_edges()
        removed = self.removed_edges()
        _all = cat(added, removed, dim=1)
        return BunchDict(added=added, removed=removed, all=_all)

    def remove_feat(self, u: int, v: int, it: Optional[int] = None):
        self._removed_feats[(u, v)] = it

    def add_feat(self, u: int, v: int, it: Optional[int] = None):
        self._added_feats[(u, v)] = it

    def removed_feats(self) -> Optional[Tensor]:
        feats = self._removed_feats
        if feats is None or len(feats) == 0:
            return None

        if isinstance(feats, dict):
            feats = list(feats.keys())

        if torch.is_tensor(feats):
            return feats.to(self.device)

        return torch.tensor(np.asarray(feats, dtype="int64").T, device=self.device)

    def added_feats(self) -> Optional[Tensor]:
        feats = self._added_feats
        if feats is None or len(feats) == 0:
            return None

        if isinstance(feats, dict):
            feats = list(feats.keys())

        if torch.is_tensor(feats):
            return feats.to(self.device)

        return torch.tensor(np.asarray(feats, dtype="int64").T, device=self.device)

    def feat_flips(self) -> BunchDict:
        added = self.added_feats()
        removed = self.removed_feats()
        _all = cat(added, removed, dim=1)
        return BunchDict(added=added, removed=removed, all=_all)

    @lru_cache(maxsize=1)
    def g(self, symmetric: bool = True) -> dgl.DGLGraph:
        """return the attacked graph

        Parameters
        ----------
        symmetric : bool
            Determine whether the resulting graph is forcibly symmetric

        Returns
        -------
        dgl.DGLGraph
            the attacked graph
        """
        graph = self.graph.local_var()

        edge_flips = self.edge_flips()
        removed = edge_flips['removed']

        if removed is not None:
            if symmetric:
                removed = torch.cat([removed, removed[[1, 0]]], dim=1)
            mask = graph.has_edges_between(removed[0], removed[1])
            e_id = graph.edge_ids(removed[0][mask], removed[1][mask])
            graph.remove_edges(e_id)

        added = edge_flips['added']
        if added is not None:
            if symmetric:
                added = torch.cat([added, added[[1, 0]]], dim=1)
            graph.add_edges(added[0], added[1])

        if self.feature_attack:
            feat = self.feat.detach().clone()
            feat_flips = self.feat_flips()
            removed = feat_flips['removed']
            if removed is not None:
                feat[removed[0], removed[1]] = 0.

            added = feat_flips['added']
            if added is not None:
                feat[added[0], added[1]] = 1.
                graph.ndata['feat'] = feat

        return graph

    def set_allow_singleton(self, state: bool):
        """Set allow_singleton flag.

        Parameters
        ----------
        state : bool
            The value to be set to the flag.
        """
        self._allow_singleton = state

    def set_allow_structure_attack(self, state: bool):
        """Set allow_structure_attack flag.

        Parameters
        ----------
        state : bool
            The value to be set to the flag.
        """
        self._allow_structure_attack = state

    def set_allow_feature_attack(self, state: bool):
        """Set allow_feature_attack flag.

        Parameters
        ----------
        state : bool
            The value to be set to the flag.
        """
        self._allow_feature_attack = state

    def is_singleton_edge(self, u: int, v: int) -> bool:
        """check if the edge is an sigleton edge that, if removed,
        would result in a sigleton node in the graph.

        Note
        ----
        please make sure the edge is the one being removed.

        Parameters
        ----------
        u : int
            the src node of the edge
        v : int
            the dst node of the edge
        Returns
        -------
        bool
            True if the edge is an singleton edge, otherwise False.
        """
        threshold = 1
        # threshold=2 if the graph has selfloop before otherwise threshold=1
        if not self._allow_singleton and (self.degree[u] <= threshold or self.degree[v] <= threshold):
            return True
        return False

    def is_legal_edge(self, u: int, v: int) -> bool:
        """check whether the edge (u,v) is legal.

        An edge (u,v) is legal if u!=v and edge (u,v) is not selected before.

        Parameters
        ----------
        u : int
            src node id
        v : int
            dst node id

        Returns
        -------
        bool
            True if the u!=v and edge (u,v) is not selected, otherwise False.
        """
        _removed_edges = self._removed_edges
        _added_edges = self._added_edges

        return all((u != v,
                    (u, v) not in _removed_edges,
                    (v, u) not in _removed_edges,
                    (u, v) not in _added_edges,
                    (v, u) not in _added_edges))


def cat(a, b, dim=1):
    if a is None:
        return b
    if b is None:
        return a

    return torch.cat([a, b], dim=dim)
