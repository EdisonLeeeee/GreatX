import warnings
import numpy as np
import torch

from functools import lru_cache
from typing import Optional, Union
from copy import copy
from torch import Tensor
from torch_geometric.data import Data

from graphwar.attack.attacker import Attacker
from graphwar.utils import BunchDict, remove_edges, add_edges


class FlipAttacker(Attacker):
    """Adversarial attacker for graph data by flipping edges.

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

    Note
    ----
    :class:`graphwar.attack.FlipAttacker` is a base class for graph modification attacks (GMA).
    """

    def reset(self) -> "FlipAttacker":
        """Reset attacker. This method must be called before attack."""
        super().reset()
        self.data.cache_clear()
        self._removed_edges = {}
        self._added_edges = {}
        self._removed_feats = {}
        self._added_feats = {}
        self.degree = self._degree.clone()
        return self

    def remove_edge(self, u: int, v: int, it: Optional[int] = None):
        """Remove an edge from the graph.

        Parameters
        ----------
        u : int
            The source node of the edge
        v : int
            The destination node of the edge
        it : Optional[int], optional
             The iteration that indicates the order of 
             the edge being removed, by default None
        """

        if not self._allow_singleton:
            is_singleton_u = self.degree[u] <= 1
            is_singleton_v = self.degree[v] <= 1

            if is_singleton_u or is_singleton_v:
                warnings.warn(f"You are trying to remove an edge ({u}-{v}) that would result in singleton nodes. "
                              "If the behavior is not intended, please make sure you have set `attacker.set_allow_singleton(False)` or check your algorithm.", UserWarning)

        self._removed_edges[(u, v)] = it
        self.degree[u] -= 1
        self.degree[v] -= 1

    def add_edge(self, u: int, v: int, it: Optional[int] = None):
        """Add one edge to the graph.

        Parameters
        ----------
        u : int
            The source node of the edge
        v : int
            The destination node of the edge
        it : Optional[int], optional
             The iteration that indicates the order of 
             the edge being added, by default None
        """
        self._added_edges[(u, v)] = it
        self.degree[u] += 1
        self.degree[v] += 1

    def removed_edges(self) -> Optional[Tensor]:
        """Get all the edges to be removed.
        """
        edges = self._removed_edges
        if edges is None or len(edges) == 0:
            return None

        if torch.is_tensor(edges):
            return edges.to(self.device)

        if isinstance(edges, dict):
            edges = list(edges.keys())

        removed = torch.tensor(np.asarray(
            edges, dtype="int64").T, device=self.device)
        return removed

    def added_edges(self) -> Optional[Tensor]:
        """Get all the edges to be added."""
        edges = self._added_edges
        if edges is None or len(edges) == 0:
            return None

        if torch.is_tensor(edges):
            return edges.to(self.device)

        if isinstance(edges, dict):
            edges = list(edges.keys())

        return torch.tensor(np.asarray(edges, dtype="int64").T, device=self.device)

    def edge_flips(self) -> BunchDict:
        """Get all the edges to be flipped, including edges to be added and removed."""
        added = self.added_edges()
        removed = self.removed_edges()
        _all = cat(added, removed, dim=1)
        return BunchDict(added=added, removed=removed, all=_all)

    def remove_feat(self, u: int, v: int, it: Optional[int] = None):
        """Remove the feature in a dimension `v` form a node `u`.
        That is, set a dimension of the specific node to zero.

        Parameters
        ----------
        u : int
            the node whose features are to be removed
        v : int
            the dimension of the feature to be removed
        it : Optional[int], optional
            The iteration that indicates the order 
            of the features being removed, by default None
        """

        self._removed_feats[(u, v)] = it

    def add_feat(self, u: int, v: int, it: Optional[int] = None):
        """Remove the feature in a dimension `v` form a node `u`.
        That is, set a dimension of the specific node to one.

        Parameters
        ----------
        u : int
            the node whose features are to be added
        v : int
            the dimension of the feature to be added
        it : Optional[int], optional
            The iteration that indicates the order 
            of the features being added, by default None
        """
        self._added_feats[(u, v)] = it

    def removed_feats(self) -> Optional[Tensor]:
        """Get all the features to be removed."""
        feats = self._removed_feats
        if feats is None or len(feats) == 0:
            return None

        if isinstance(feats, dict):
            feats = list(feats.keys())

        if torch.is_tensor(feats):
            return feats.to(self.device)

        return torch.tensor(np.asarray(feats, dtype="int64").T, device=self.device)

    def added_feats(self) -> Optional[Tensor]:
        """Get all the features to be added."""
        feats = self._added_feats
        if feats is None or len(feats) == 0:
            return None

        if isinstance(feats, dict):
            feats = list(feats.keys())

        if torch.is_tensor(feats):
            return feats.to(self.device)

        return torch.tensor(np.asarray(feats, dtype="int64").T, device=self.device)

    def feat_flips(self) -> BunchDict:
        """Get all the features to be flipped, including features to be added and removed."""
        added = self.added_feats()
        removed = self.removed_feats()
        _all = cat(added, removed, dim=1)
        return BunchDict(added=added, removed=removed, all=_all)

    @lru_cache(maxsize=1)
    def data(self, symmetric: bool = True) -> Data:
        """Get the attacked graph denoted by
        PyG-like data instance.

        Parameters
        ----------
        symmetric : bool, optional
            whether the output graph is symmetric, by default True

        Returns
        -------
        Data
            the attacked graph denoted by PyG-like data instance
        """

        data = copy(self.ori_data)
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        assert edge_weight is None, 'weighted graph is not supported now.'

        edge_flips = self.edge_flips()
        removed = edge_flips['removed']

        if removed is not None:
            edge_index = remove_edges(edge_index, removed, symmetric=symmetric)

        added = edge_flips['added']
        if added is not None:
            edge_index = add_edges(edge_index, added, symmetric=symmetric)

        data.edge_index = edge_index

        if edge_weight is not None:
            data.edge_weight = edge_weight

        if self.feature_attack:
            feat = self.feat.detach().clone()
            feat_flips = self.feat_flips()
            removed = feat_flips['removed']
            if removed is not None:
                feat[removed[0], removed[1]] = 0.

            added = feat_flips['added']
            if added is not None:
                feat[added[0], added[1]] = 1.
            data.x = feat

        return data

    def set_allow_singleton(self, state: bool):
        """Set whether the attacked graph allow singleton node, i.e., 
        zero degree nodes.

        Parameters
        ----------
        state : bool
            the flag to set

        Example
        -------
        >>> attacker.set_allow_singleton(True)            
        """

        self._allow_singleton = state

    def set_allow_structure_attack(self, state: bool):
        """Set whether the attacker allow attacks on the topology of the graph.

        Parameters
        ----------
        state : bool
            the flag to set

        Example
        -------
        >>> attacker.set_allow_structure_attack(True)                  
        """
        self._allow_structure_attack = state

    def set_allow_feature_attack(self, state: bool):
        """Set whether the attacker allow attacks on the features of nodes in the graph. 

        Parameters
        ----------
        state : bool
            the flag to set

        Example
        -------
        >>> attacker.set_allow_feature_attack(True)                
        """
        self._allow_feature_attack = state

    def is_singleton_edge(self, u: int, v: int) -> bool:
        """Check if the edge is an singleton edge that, if removed,
        would result in a singleton node in the graph.

        Parameters
        -----------
        u : int
            The source node of the edge
        v : int
            The destination node of the edge

        Return
        ------
        bool: `True` if the edge is an singleton edge, otherwise `False`.

        Note
        ----
        Please make sure the edge is the one being removed.        
        """
        threshold = 1
        # threshold = 2 if the graph has selfloop before otherwise threshold = 1
        if not self._allow_singleton and (self.degree[u] <= threshold or self.degree[v] <= threshold):
            return True
        return False

    def is_legal_edge(self, u: int, v: int) -> bool:
        """Check whether the edge (u,v) is legal.

        An edge (u,v) is legal if u!=v and edge (u,v) is not selected before.

        Parameters
        -----------
        u : int
            The source node of the edge
        v : int
            The destination node of the edge

        Returns
        -------
        bool: `True` if the u!=v and edge (u,v), (v,u) is not selected, otherwise `False`.
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
