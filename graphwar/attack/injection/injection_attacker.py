from functools import lru_cache
from typing import Optional, Union

from copy import copy
import torch
import numpy as np
from torch import Tensor

from graphwar.attack.attacker import Attacker
from graphwar.utils import add_edges
from torch_geometric.data import Data


class InjectionAttacker(Attacker):

    def reset(self) -> "InjectionAttacker":
        """Reset the state of the Attacker

        Returns
        -------
        InjectionAttacker
            the attacker itself
        """
        super().reset()
        self.num_budgets = None
        self.feat_limits = None
        self.num_edges_global = None
        self.num_edges_local = None
        self._injected_nodes = []
        self._injected_edges = {}
        self._injected_feats = []
        self.data.cache_clear()

        return self

    def attack(self, num_budgets: Union[int, float], *, targets: Optional[Tensor] = None, num_edges_global: Optional[int] = None,
               num_edges_local: Optional[int] = None, feat_limits: Optional[Union[tuple, dict]] = None) -> "InjectionAttacker":
        """Base method that describes the adversarial injection attack
        """

        _is_setup = getattr(self, "_is_setup", True)

        if not _is_setup:
            raise RuntimeError(
                f'{self.name} requires a surrogate model to conduct attack. '
                'Use `attacker.setup_surrogate(surrogate_model)`.')

        if not self._is_reset:
            raise RuntimeError(
                'Before calling attack, you must reset your attacker. Use `attacker.reset()`.'
            )

        num_budgets = self._check_budget(
            num_budgets, max_perturbations=self.num_nodes)

        assert num_edges_global is None, 'Not implemented now!'

        if num_edges_local is None:
            num_edges_local = int(self._degree.mean().clamp(min=1))

        self.num_budgets = num_budgets
        self.num_edges_global = num_edges_global
        self.num_edges_local = num_edges_local

        # ============== get feature limitation of injected node ==============
        min_limits = max_limits = None

        if feat_limits is not None:
            if isinstance(feat_limits, tuple):
                min_limits, max_limits = feat_limits
            elif isinstance(feat_limits, dict):
                min_limits = feat_limits.pop('min', None)
                max_limits = feat_limits.pop('max', None)
                if feat_limits:
                    raise ValueError(
                        f"Unrecognized key {next(iter(feat_limits.keys()))}.")
            else:
                raise TypeError(
                    f"`feat_limits` should be an instance of tuple and dict, but got {feat_limits}.")

        feat = self.feat

        if min_limits is None and feat is not None:
            min_limits = feat.min()
        else:
            min_limits = 0.

        if max_limits is None and feat is not None:
            max_limits = feat.max()
        else:
            max_limits = 1.

        self._mu = (max_limits - min_limits) / 2
        self._sigma = (max_limits - self._mu) / 3  # 3-sigma rule
        # ======================================================================

        self.feat_limits = min_limits, max_limits
        if targets is None:
            self.targets = list(range(self.num_nodes))
        else:
            self.targets = torch.LongTensor(targets).view(-1).tolist()
        self._is_reset = False

        return self

    def injected_nodes(self) -> Optional[Tensor]:
        """Get all the nodes to be injected."""
        nodes = self._injected_nodes
        if nodes is None or len(nodes) == 0:
            return None

        if torch.is_tensor(nodes):
            return nodes.to(self.device)

        if isinstance(nodes, dict):
            nodes = sorted(list(nodes.keys()))

        return torch.tensor(np.asarray(nodes, dtype="int64"), device=self.device)

    def added_nodes(self) -> Optional[Tensor]:
        """alias of method `added_nodes`"""
        return self.injected_nodes()

    def injected_edges(self) -> Optional[Tensor]:
        """Get all the edges to be injected."""
        edges = self._injected_edges
        if edges is None or len(edges) == 0:
            return None

        if torch.is_tensor(edges):
            return edges.to(self.device)

        if isinstance(edges, dict):
            edges = list(edges.keys())

        return torch.tensor(np.asarray(edges, dtype="int64").T, device=self.device)

    def added_edges(self) -> Optional[Tensor]:
        """alias of method `injected_edges`"""
        return self.injected_edges()

    def injected_feats(self) -> Optional[Tensor]:
        """Get the features injected nodes."""
        feats = self._injected_feats
        if feats is None or len(feats) == 0:
            return None
        # feats = list(self._injected_nodes.values())
        return torch.stack(feats, dim=0).float().to(self.device)

    def added_feats(self) -> Optional[Tensor]:
        """alias of method `added_edges`"""
        return self.injected_feats()

    def inject_node(self, node, feat: Optional[Tensor] = None):
        if feat is None:
            feat = self.feat.new_empty(
                self.num_feats).uniform_(*self.feat_limits)
        else:
            assert feat.min() >= self.feat_limits[0]
            assert feat.max() <= self.feat_limits[1]

        self._injected_nodes.append(node)
        self._injected_feats.append(feat)

    def inject_edge(self, u: int, v: int, it: Optional[int] = None):
        """Inject one edge to the graph.

        Parameters
        ----------
        u : int
             The source node of the edge.
        v : int
            The destination node of the edge.
        it : Optional[int], optional
            The iteration that indicates the order of the edge being added, by default None
        """

        self._injected_edges[(u, v)] = it

    @lru_cache(maxsize=1)
    def data(self, symmetric: bool = True) -> Data:
        """return the attacked graph

        Parameters
        ----------
        symmetric : bool
            Determine whether the resulting graph is forcibly symmetric

        Returns
        -------
        Data
            the attacked graph
        """
        data = copy(self.ori_data)
        # injected_nodes = self.injected_nodes()
        injected_edges = self.injected_edges()
        injected_feats = self.injected_feats()
        data.x = torch.cat([data.x, injected_feats], dim=0)
        data.edge_index = add_edges(
            data.edge_index, injected_edges, symmetric=symmetric)

        return data
