import warnings
from functools import lru_cache
from typing import Optional

import numpy as np
import scipy.sparse as sp
from numba import njit
from tqdm import tqdm
from torch_geometric.data import Data

from graphwar import Surrogate
from graphwar.attack.targeted.targeted_attacker import TargetedAttacker
from graphwar.utils import singleton_filter, scipy_normalize, LikelihoodFilter


class Nettack(TargetedAttacker, Surrogate):
    r"""Implementation of `Nettack` attack from the: 
    `"Adversarial Attacks on Neural Networks for Graph Data" 
    <https://arxiv.org/abs/1805.07984>`_ paper (KDD'18)

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

    >>> surrogate_model = ... # train your surrogate model

    >>> from graphwar.attack.targeted import Nettack
    >>> attacker = Nettack(data)
    >>> attacker.setup_surrogate(surrogate_model)
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

    # Nettack can conduct feature attack
    _allow_feature_attack = True
    _allow_singleton: bool = False

    def __init__(self, data: Data, device: str = "cpu",
                 seed: Optional[int] = None, name: Optional[str] = None, **kwargs):
        super().__init__(data=data, device=device, seed=seed, name=name, **kwargs)
        feat = self.feat
        self.scipy_feat = sp.csr_matrix(feat.cpu().numpy())
        self.cooc_matrix = sp.csr_matrix((feat.t() @ feat).cpu().numpy())

    def setup_surrogate(self, surrogate):
        Surrogate.setup_surrogate(self, surrogate=surrogate, freeze=True)
        W = None
        for para in self.surrogate.parameters():
            if para.ndim == 1:
                warnings.warn(f"The surrogate model has `bias` term, which is ignored and the "
                              f"model itself may not be a perfect choice for {self.name}.")
                continue
            if W is None:
                W = para
            else:
                W = para @ W

        assert W is not None
        self.W = W.t().cpu().numpy()
        self.num_classes = self.W.shape[-1]
        return self

    def reset(self):
        super().reset()
        self.modified_adj = self.adjacency_matrix.copy()
        self.modified_feat = self.scipy_feat.copy()
        self.adj_norm = scipy_normalize(self.modified_adj)
        self.cooc_constraint = None
        return self

    def compute_cooccurrence_constraint(self, nodes):
        num_nodes = self.num_nodes
        num_feats = self.num_feats
        words_graph = self.cooc_matrix - \
            sp.diags(self.cooc_matrix.diagonal(), format='csr')
        words_graph.eliminate_zeros()
        words_graph.data = words_graph.data > 0
        word_degrees = words_graph.sum(0).A1
        inv_word_degrees = np.reciprocal(word_degrees.astype(float) + 1e-8)

        sd = np.zeros(num_nodes)
        for n in range(num_nodes):
            n_idx = self.modified_feat[n].nonzero()[1]
            sd[n] = np.sum(inv_word_degrees[n_idx.tolist()])

        scores_matrix = sp.lil_matrix((num_nodes, num_feats))

        for n in nodes:
            common_words = words_graph.multiply(self.modified_feat[n])
            idegs = inv_word_degrees[common_words.nonzero()[1]]
            nnz = common_words.nonzero()[0]
            scores = np.array([idegs[nnz == ix].sum()
                              for ix in range(num_feats)])
            scores_matrix[n] = scores

        self.cooc_constraint = sp.csr_matrix(
            scores_matrix - 0.5 * sd[:, None] > 0)

    def gradient_wrt_x(self, label):
        return (self.adj_norm @ self.adj_norm)[self.target].T @ sp.coo_matrix(self.W[:, label].reshape(1, -1))

    def compute_logits(self):
        return (self.adj_norm @ self.adj_norm @ self.modified_feat @ self.W)[self.target].ravel()

    def strongest_wrong_class(self, logits):
        target_label_onehot = np.eye(self.num_classes)[self.target_label]
        return (logits - 1000 * target_label_onehot).argmax()

    def feature_scores(self):
        if self.cooc_constraint is None:
            self.compute_cooccurrence_constraint(self.influence_nodes)

        logits = self.compute_logits()
        best_wrong_class = self.strongest_wrong_class(logits)
        gradient = self.gradient_wrt_x(
            self.target_label) - self.gradient_wrt_x(best_wrong_class)
        surrogate_loss = logits[self.target_label] - logits[best_wrong_class]

        gradients_flipped = (gradient * -1).tolil()
        gradients_flipped[self.modified_feat.nonzero()] *= -1

        X_influencers = sp.lil_matrix(self.modified_feat.shape)
        X_influencers[self.influence_nodes] = self.modified_feat[self.influence_nodes]
        gradients_flipped = gradients_flipped.multiply(
            (self.cooc_constraint + X_influencers) > 0)
        nnz_ixs = np.array(gradients_flipped.nonzero()).T

        sorting = np.argsort(gradients_flipped[tuple(nnz_ixs.T)]).A1
        sorted_ixs = nnz_ixs[sorting]
        grads = gradients_flipped[tuple(nnz_ixs[sorting].T)]

        scores = surrogate_loss - grads
        return sorted_ixs[::-1], scores.A1[::-1]

    def structure_score(self, a_hat_uv, XW):
        logits = a_hat_uv @ XW
        label_onehot = np.eye(self.num_classes)[self.target_label]
        best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
        logits_for_correct_class = logits[:, self.target_label]
        struct_scores = logits_for_correct_class - best_wrong_class_logits

        return struct_scores

    @lru_cache(maxsize=1)
    def compute_XW(self):
        return self.modified_feat @ self.W

    def get_attacker_nodes(self, n=5, add_additional_nodes=False):

        assert n < self.modified_adj.shape[0] - \
            1, "number of influencers cannot be >= number of nodes in the graph!"
        neighbors = self.modified_adj[self.target].indices
        candidate_edges = np.column_stack(
            (np.tile(self.target, len(neighbors)), neighbors)).astype("int32")
        # The new A_hat_square_uv values that we would get if we removed the edge from u to each of the neighbors, respectively
        a_hat_uv = self.compute_new_a_hat_uv(candidate_edges)

        XW = self.compute_XW()

        # compute the struct scores for all neighbors
        struct_scores = self.structure_score(a_hat_uv, XW)
        if len(neighbors) >= n:  # do we have enough neighbors for the number of desired influencers?
            influence_nodes = neighbors[np.argsort(struct_scores)[:n]]
            if add_additional_nodes:
                return influence_nodes, np.array([])
            return influence_nodes
        else:
            influence_nodes = neighbors
            if add_additional_nodes:  # Add additional influencers by connecting them to u first.
                # Compute the set of possible additional influencers, i.e. all nodes except the ones
                # that are already connected to u.
                poss_add_infl = np.setdiff1d(np.setdiff1d(
                    np.arange(self.modified_adj.shape[0]), neighbors), self.target)
                n_possible_additional = len(poss_add_infl)
                n_additional_attackers = n - len(neighbors)
                possible_edges = np.column_stack(
                    (np.tile(self.target, n_possible_additional), poss_add_infl)).astype("int32")

                # Compute the struct_scores for all possible additional influencers, and choose the one
                # with the best struct score.
                a_hat_uv_additional = self.compute_new_a_hat_uv(possible_edges)
                additional_struct_scores = self.structure_score(
                    a_hat_uv_additional, XW)
                # TODO: is it right?
                additional_influencers = poss_add_infl[np.argsort(
                    additional_struct_scores)[-n_additional_attackers::]]

                return influence_nodes, additional_influencers
            else:
                return influence_nodes

    def compute_new_a_hat_uv(self, candidate_edges):

        edges = np.transpose(self.modified_adj.nonzero())
        edges_set = {tuple(e) for e in edges}
        A_hat_sq = self.adj_norm @ self.adj_norm
        values_before = A_hat_sq[self.target].toarray()[0]
        node_ixs = np.unique(edges[:, 0], return_index=True)[1].astype("int32")
        twohop_ixs = np.transpose(A_hat_sq.nonzero())
        degrees = self.modified_adj.sum(0).A1 + 1

        # Ignore warnings:
        #     NumbaPendingDeprecationWarning:
        # Encountered the use of a type that is scheduled for deprecation: type 'reflected set' found for argument 'edges_set' of function 'compute_new_a_hat_uv'.
        # For more information please refer to http://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings(
                'ignore',
                '.*Encountered the use of a type that is scheduled for deprecation*'
            )
            ixs, vals = compute_new_a_hat_uv(edges, node_ixs, edges_set,
                                             twohop_ixs, values_before,
                                             degrees, candidate_edges,
                                             self.target)
        ixs_arr = np.array(ixs)
        a_hat_uv = sp.coo_matrix((vals, (ixs_arr[:, 0], ixs_arr[:, 1])),
                                 shape=[len(candidate_edges), self.modified_adj.shape[0]])

        return a_hat_uv

    def get_candidate_edges(self, n_influencers):
        # Potential edges are all edges from any attacker to any other node, except the respective
        # attacker itself or the node being attacked.
        target = self.target
        N = self.num_nodes
        nodes_set = set(range(N)) - set([target])

        if self.direct_attack:
            influencers = [target]
            row = np.repeat(influencers, N - 1)
            col = list(nodes_set)
        else:
            infls, add_infls = self.get_attacker_nodes(
                n_influencers, add_additional_nodes=True)
            influencers = np.concatenate((infls, add_infls))
            # influencers = self.adjacency_matrix[target].indices
            row = np.repeat(influencers, N - 2)
            col = np.hstack([list(nodes_set - set([infl]))
                            for infl in influencers])

        candidate_edges = np.stack([row, col], axis=1)
        self.influence_nodes = np.asarray(influencers)
        return candidate_edges

    def attack(self,
               target, *,
               target_label=None,
               num_budgets=None,
               n_influencers=5,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               ll_constraint=True,
               ll_cutoff=0.004,
               disable=False):

        super().attack(target, target_label, num_budgets=num_budgets,
                       direct_attack=direct_attack, structure_attack=structure_attack,
                       feature_attack=feature_attack)

        if feature_attack:
            self._check_feature_matrix_binary()

        if ll_constraint and self._allow_singleton:
            raise RuntimeError(
                '`ll_constraint` is failed when `allow_singleton=True`, please set `attacker.set_allow_singleton(False)`.'
            )

        if target_label is None:
            assert self.target_label is not None, "please specify argument `target_label` as the node label does not exist."
            target_label = self.target_label.item()

        candidate_edges = self.get_candidate_edges(
            n_influencers).astype("int32")

        if ll_constraint:
            likelihood_filter = LikelihoodFilter(self.degree.cpu().numpy(),
                                                 ll_cutoff=ll_cutoff)

        for it in tqdm(range(self.num_budgets),
                       desc='Perturbing graph...',
                       disable=disable):

            best_edge_score = best_feature_score = 0
            if structure_attack:
                # Do not consider edges that, if removed, result in singleton edges in the graph.
                if not self._allow_singleton:
                    candidate_edges = singleton_filter(
                        candidate_edges, self.modified_adj)

                if ll_constraint:
                    # Do not consider edges that, if removed, result in singleton edges in the graph.
                    candidate_edges = likelihood_filter(candidate_edges,
                                                        edge_weights=self.modified_adj[candidate_edges[:, 0],
                                                                                       candidate_edges[:, 1]].A1)

                # Compute new entries in A_hat_square_uv
                a_hat_uv_new = self.compute_new_a_hat_uv(candidate_edges)
                # Compute the struct scores for each potential edge
                struct_scores = self.structure_score(
                    a_hat_uv_new, self.compute_XW())
                best_edge_ix = struct_scores.argmin()
                best_edge_score = struct_scores.min()
                best_edge = candidate_edges[best_edge_ix]

            if feature_attack:
                # Compute the feature scores for each potential feature perturbation
                feature_ixs, feature_scores = self.feature_scores()
                best_feat = feature_ixs[0]
                best_feature_score = feature_scores[0]

            if structure_attack and feature_attack:
                # decide whether to choose an edge or feature to change
                if best_edge_score < best_feature_score:
                    change_structure = True
                else:
                    change_structure = False

            elif structure_attack:
                change_structure = True
            elif feature_attack:
                change_structure = False

            if change_structure:
                # perform edge perturbation
                u, v = best_edge
                edge_weight = self.modified_adj[(u, v)]
                modified_adj = self.modified_adj.tolil(copy=False)
                modified_adj[(u, v)] = modified_adj[(
                    v, u)] = 1 - modified_adj[(u, v)]
                self.modified_adj = modified_adj.tocsr(copy=False)
                self.adj_norm = scipy_normalize(self.modified_adj)
                if edge_weight > 0:
                    self.remove_edge(u, v, it)
                else:
                    self.add_edge(u, v, it)
                np.delete(candidate_edges, best_edge_ix, axis=0)

                if ll_constraint:
                    # Update likelihood ratio test values
                    likelihood_filter.update(u, v, edge_weight, best_edge_ix)
            else:
                u, v = best_feat
                feat_weight = self.modified_feat[(u, v)]
                modified_feat = self.modified_feat.tolil(copy=False)
                modified_feat[(u, v)] = 1 - modified_feat[(u, v)]
                self.modified_feat = modified_feat.tocsr(copy=False)
                if feat_weight > 0:
                    self.remove_feat(u, v, it)
                else:
                    self.add_feat(u, v, it)
                self.compute_XW.cache_clear()
        return self


@ njit
def connected_after(u, v, connected_before, delta):
    if u == v:
        if delta == -1:
            return False
        else:
            return True
    else:
        return connected_before


@ njit
def compute_new_a_hat_uv(edge_ixs, node_nb_ixs, edges_set, twohop_ixs,
                         values_before, degs, candidate_edges, u):
    """
    Compute the new values [A_hat_square]_u for every potential edge, where u is the target node. C.f. Theorem 5.1
    equation 17.

    Parameters
    ----------
    edge_ixs: np.array, shape [E,2], where E is the number of edges in the graph.
        The indices of the nodes connected by the edges in the input graph.
    node_nb_ixs: np.array, shape [num_nodes,], dtype int
        For each node, this gives the first index of edges associated to this node in the edge array (edge_ixs).
        This will be used to quickly look up the neighbors of a node, since numba does not allow nested lists.
    edges_set: set((e0, e1))
        The set of edges in the input graph, i.e. e0 and e1 are two nodes connected by an edge
    twohop_ixs: np.array, shape [T, 2], where T is the number of edges in A_tilde^2
        The indices of nodes that are in the twohop neighborhood of each other, including self-loops.
    values_before: np.array, shape [num_nodes,], the values in [A_hat]^2_uv to be updated.
    degs: np.array, shape [num_nodes,], dtype int
        The degree of the nodes in the input graph.
    candidate_edges: np.array, shape [P, 2], where P is the number of potential edges.
        The potential edges to be evaluated. For each of these potential edges, this function will compute the values
        in [A_hat]^2_uv that would result after inserting/removing this edge.
    u: int
        The target node

    Returns
    -------
    return_ixs: List of tuples
        The ixs in the [P, num_nodes] matrix of updated values that have changed
    return_values:

    """
    num_nodes = degs.shape[0]

    twohop_u = twohop_ixs[twohop_ixs[:, 0] == u, 1]
    nbs_u = edge_ixs[edge_ixs[:, 0] == u, 1]
    nbs_u_set = set(nbs_u)

    return_ixs = []
    return_values = []

    for ix in range(len(candidate_edges)):
        edge = candidate_edges[ix]
        edge_set = set(edge)
        degs_new = degs.copy()
        delta = -2 * ((edge[0], edge[1]) in edges_set) + 1
        degs_new[edge] += delta

        nbs_edge0 = edge_ixs[edge_ixs[:, 0] == edge[0], 1]
        nbs_edge1 = edge_ixs[edge_ixs[:, 0] == edge[1], 1]

        affected_nodes = set(np.concatenate((twohop_u, nbs_edge0, nbs_edge1)))
        affected_nodes = affected_nodes.union(edge_set)
        a_um = edge[0] in nbs_u_set
        a_un = edge[1] in nbs_u_set

        a_un_after = connected_after(u, edge[0], a_un, delta)
        a_um_after = connected_after(u, edge[1], a_um, delta)

        for v in affected_nodes:
            a_uv_before = v in nbs_u_set
            a_uv_before_sl = a_uv_before or v == u

            if v in edge_set and u in edge_set and u != v:
                if delta == -1:
                    a_uv_after = False
                else:
                    a_uv_after = True
            else:
                a_uv_after = a_uv_before
            a_uv_after_sl = a_uv_after or v == u

            from_ix = node_nb_ixs[v]
            to_ix = node_nb_ixs[v + 1] if v < num_nodes - 1 else len(edge_ixs)
            node_nbs = edge_ixs[from_ix:to_ix, 1]
            node_nbs_set = set(node_nbs)
            a_vm_before = edge[0] in node_nbs_set

            a_vn_before = edge[1] in node_nbs_set
            a_vn_after = connected_after(v, edge[0], a_vn_before, delta)
            a_vm_after = connected_after(v, edge[1], a_vm_before, delta)

            mult_term = 1. / np.sqrt(degs_new[u] * degs_new[v])

            sum_term1 = np.sqrt(degs[u] * degs[v]) * values_before[v] - a_uv_before_sl / degs[u] - a_uv_before / \
                degs[v]
            sum_term2 = a_uv_after / degs_new[v] + a_uv_after_sl / degs_new[u]
            sum_term3 = -((a_um and a_vm_before) / degs[edge[0]]) + (
                a_um_after and a_vm_after) / degs_new[edge[0]]
            sum_term4 = -((a_un and a_vn_before) / degs[edge[1]]) + (
                a_un_after and a_vn_after) / degs_new[edge[1]]
            new_val = mult_term * (sum_term1 + sum_term2 + sum_term3 +
                                   sum_term4)

            return_ixs.append((ix, v))
            return_values.append(new_val)

    return return_ixs, return_values
