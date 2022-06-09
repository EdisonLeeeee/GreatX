import warnings

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor

__all__ = ["singleton_filter", "SingletonFilter",
           "LikelihoodFilter", "singleton_mask"]


def singleton_filter(edges: np.ndarray, adj_matrix: sp.csr_matrix):
    """Filter edges that, if removed, would turn one or more nodes 
    into singleton nodes.

    Parameters
    ----------
    edges: np.array, shape [M, 2], where M is the number of input edges.
        The candidate edges to remove.
    adj_matrix: sp.sparse_matrix, shape [num_nodes, num_nodes]
        The input adjacency matrix where edges derived from.

    Returns
    -------
    np.array, shape [M, 2], 
        the edges that removed will not generate singleton nodes.
    """
    assert edges.shape[1] == 2, f"edges should be shape [M, 2], bug got {edges.shape}"
    if len(edges) == 0:
        warnings.warn("No edges found.", RuntimeWarning)
        return edges

    deg = adj_matrix.sum(1).A1
    existing_edge = adj_matrix.tocsr(copy=False)[edges[:, 0], edges[:, 1]].A1

    if existing_edge.size > 0:
        edge_degrees = deg[edges] - 2 * existing_edge[:, None] + 1
    else:
        edge_degrees = deg[edges] + 1

    mask = np.logical_and(edge_degrees[:, 0] > 0, edge_degrees[:, 1] > 0)
    return edges[mask]


def singleton_mask(adj_matrix: Tensor):
    """Computes a mask for entries potentially 
    leading to singleton nodes, i.e. one of the 
    two nodes corresponding to the entry have 
    degree 1 and there is an edge between the two nodes.

    Parameters
    ----------
    adj_matrix : Tensor, shape [N, N], 
        where N is the number of nodes
        the input adjacency matrix to compte the mask

    Returns
    -------
    mask : bool Tensor
        a boolean mask with shape as :obj:`adj_matrix`.
    """

    N = adj_matrix.size(0)
    degrees = adj_matrix.sum(1)
    degree_one = degrees == 1
    resh = degree_one.repeat(N).view(N, N)
    l_and = torch.logical_and(resh, adj_matrix > 0)
    logical_and_symmetric = torch.logical_or(l_and, l_and.t())
    mask = 1. - logical_and_symmetric.float()
    return mask


class SingletonFilter:
    """Computes a mask for entries potentially 
    leading to singleton nodes, i.e. one of the 
    two nodes corresponding to the entry have 
    degree 1 and there is an edge between the two nodes.

    Parameters
    ----------
    adj_matrix : sp.csr_matrix
        the input adjacency matrix    
    """

    def __init__(self, adj_matrix: sp.csr_matrix):
        self.degree = adj_matrix.sum(1).A1

    def __call__(self, edges: np.ndarray, adj_matrix: sp.csr_matrix):
        return singleton_filter(edges, adj_matrix)

    def update(self, u: int, v: int, edge_weight: float):
        delta = 1 - 2 * edge_weight
        self.degree[u] += delta
        self.degree[v] += delta


class LikelihoodFilter:
    """Likelihood filter from the 
    `"Adversarial Attacks on Neural Networks for Graph Data" 
    <https://arxiv.org/abs/1805.07984>`_ paper (KDD'18)

    Parameters
    ----------
    degree : np.ndarray
        the degree of the nodes in the graph
    ll_cutoff : float, optional
        likelihood cutoff, by default 0.004    
    """

    def __init__(self, degree: np.ndarray, ll_cutoff: float = 0.004):

        self.ll_cutoff = ll_cutoff

        # Setup starting values of the likelihood ratio test.
        degree_sequence_start = degree

        d_min = 2  # denotes the minimum degree a node needs to have to be considered in the power-law test
        S_d_start = np.sum(
            np.log(degree_sequence_start[degree_sequence_start >= d_min]))
        n_start = np.sum(degree_sequence_start >= d_min)
        alpha_start = self.compute_alpha(n_start, S_d_start, d_min)

        self.log_likelihood_start = self.compute_log_likelihood(
            n_start, alpha_start, S_d_start, d_min)
        self.S_d_start = S_d_start
        self.current_S_d = S_d_start.copy()
        self.n_start = n_start
        self.current_n = n_start.copy()
        self.current_degree_sequence = degree_sequence_start.copy()
        self.d_min = d_min

    def __call__(self, edges: np.ndarray, edge_weights: np.ndarray) -> np.ndarray:
        """Do not consider edges that, if added/removed, would lead to a violation of the
            likelihood ration Chi_square cutoff value.
        """
        n_start = self.n_start
        S_d_start = self.S_d_start
        current_S_d = self.current_S_d
        current_n = self.current_n
        d_min = self.d_min

        # Update the values for the power law likelihood ratio test.
        deltas = 1 - 2 * edge_weights
        d_edges_old = self.current_degree_sequence[edges]
        d_edges_new = self.current_degree_sequence[edges] + deltas[:, None]
        new_S_d, new_n = self.update_Sx(
            current_S_d, current_n, d_edges_old, d_edges_new, d_min)
        new_alphas = self.compute_alpha(new_n, new_S_d, d_min)
        new_ll = self.compute_log_likelihood(new_n, new_alphas, new_S_d, d_min)
        alphas_combined = self.compute_alpha(
            new_n + n_start, new_S_d + S_d_start, d_min)
        new_ll_combined = self.compute_log_likelihood(
            new_n + n_start, alphas_combined, new_S_d + S_d_start, d_min)
        new_ratios = -2 * new_ll_combined + 2 * \
            (new_ll + self.log_likelihood_start)
        mask = self.filter_chisquare(new_ratios, self.ll_cutoff)

        self.new_S_d = new_S_d[mask]
        self.new_n = new_n[mask]
        return edges[mask]

    def update(self, u: int, v: int, edge_weight: float, idx: int):
        """Update likelihood ratio test values
        """
        delta = 1 - 2 * edge_weight
        self.current_S_d = self.new_S_d[idx]
        self.current_n = self.new_n[idx]
        self.current_degree_sequence[[u, v]] += delta

    @staticmethod
    def compute_alpha(n, S_d, d_min):
        """Approximate the alpha of a power law distribution.

        Parameters
        ----------
        n: int or np.array of int
            Number of entries that are larger than or equal to d_min
        S_d: float or np.array of float
            Sum of log degrees in the distribution that are larger than or equal to d_min
        d_min: int
            The minimum degree of nodes to consider

        Returns
        -------
        alpha: float
            The estimated alpha of the power law distribution
        """

        return n / (S_d - n * np.log(d_min - 0.5)) + 1

    @staticmethod
    def update_Sx(S_old, n_old, d_old, d_new, d_min):
        """Update on the sum of log degrees S_d and n based on degree distribution resulting from inserting or deleting
        a single edges.

        Parameters
        ----------
        S_old: float
            Sum of log degrees in the distribution that are larger than or equal to d_min.
        n_old: int
            Number of entries in the old distribution that are larger than or equal to d_min.
        d_old: np.array, shape [num_nodes,] dtype int
            The old degree sequence.
        d_new: np.array, shape [num_nodes,] dtype int
            The new degree sequence
        d_min: int
            The minimum degree of nodes to consider

        Returns
        -------
        new_S_d: float, the updated sum of log degrees in the distribution that are larger than or equal to d_min.
        new_n: int, the updated number of entries in the old distribution that are larger than or equal to d_min.
        """

        old_in_range = d_old >= d_min
        new_in_range = d_new >= d_min

        d_old_in_range = np.multiply(d_old, old_in_range)
        d_new_in_range = np.multiply(d_new, new_in_range)

        new_S_d = S_old - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(
            np.maximum(d_new_in_range, 1)).sum(1)
        new_n = n_old - np.sum(old_in_range, 1) + np.sum(new_in_range, 1)

        return new_S_d, new_n

    @staticmethod
    def compute_log_likelihood(n, alpha, S_d, d_min):
        """Compute log likelihood of the powerlaw fit.

        Parameters
        ----------
        n: int
            Number of entries in the old distribution that are larger than or equal to d_min.
        alpha: float
            The estimated alpha of the power law distribution
        S_d: float
            Sum of log degrees in the distribution that are larger than or equal to d_min.
        d_min: int
            The minimum degree of nodes to consider

        Returns
        -------
        float: the estimated log likelihood
        """

        return n * np.log(alpha) + n * alpha * np.log(d_min) - (alpha + 1) * S_d

    @staticmethod
    def filter_chisquare(ll_ratios, cutoff):
        return ll_ratios < cutoff


class LikelihoodFilterTensor:
    """Likelihood filter (Tensor version) )from the 
    `"Adversarial Attacks on Neural Networks for Graph Data" 
    <https://arxiv.org/abs/1805.07984>`_ paper (KDD'18)

    Parameters
    ----------
    degree : Tensor
        the degree of the nodes in the graph
    ll_cutoff : float, optional
        likelihood cutoff, by default 0.004       
    """

    def __init__(self, degree: torch.Tensor, ll_cutoff: float = 0.004):

        self.ll_cutoff = ll_cutoff

        # Setup starting values of the likelihood ratio test.
        degree_sequence_start = degree

        # denotes the minimum degree a node needs to have to be considered in the power-law test
        d_min = torch.as_tensor(2.0).to(degree)
        S_d_start = torch.sum(
            torch.log(degree_sequence_start[degree_sequence_start >= d_min]))
        n_start = torch.sum(degree_sequence_start >= d_min)
        alpha_start = self.compute_alpha(n_start, S_d_start, d_min)

        self.log_likelihood_start = self.compute_log_likelihood(
            n_start, alpha_start, S_d_start, d_min)
        self.S_d_start = S_d_start
        self.current_S_d = S_d_start.clone()
        self.n_start = n_start
        self.current_n = n_start.clone()
        self.current_degree_sequence = degree_sequence_start.clone()
        self.d_min = d_min

    def __call__(self, edges, edge_weights):
        """Do not consider edges that, if added/removed, would lead to a violation of the
            likelihood ration Chi_square cutoff value.
        """
        n_start = self.n_start
        S_d_start = self.S_d_start
        current_S_d = self.current_S_d
        current_n = self.current_n
        d_min = self.d_min

        # Update the values for the power law likelihood ratio test.
        deltas = 1 - 2 * edge_weights
        d_edges_old = self.current_degree_sequence[edges]
        d_edges_new = self.current_degree_sequence[edges] + deltas[:, None]
        new_S_d, new_n = self.update_Sx(
            current_S_d, current_n, d_edges_old, d_edges_new, d_min)
        new_alphas = self.compute_alpha(new_n, new_S_d, d_min)
        new_ll = self.compute_log_likelihood(new_n, new_alphas, new_S_d, d_min)
        alphas_combined = self.compute_alpha(
            new_n + n_start, new_S_d + S_d_start, d_min)
        new_ll_combined = self.compute_log_likelihood(
            new_n + n_start, alphas_combined, new_S_d + S_d_start, d_min)
        new_ratios = -2 * new_ll_combined + 2 * \
            (new_ll + self.log_likelihood_start)
        mask = self.filter_chisquare(new_ratios, self.ll_cutoff)

        self.new_S_d = new_S_d[mask]
        self.new_n = new_n[mask]
        return edges[mask]

    def update(self, u, v, edge_weight, idx):
        """Update likelihood ratio test values
        """
        delta = 1 - 2 * edge_weight
        self.current_S_d = self.new_S_d[idx]
        self.current_n = self.new_n[idx]
        self.current_degree_sequence[[u, v]] += delta

    @staticmethod
    def compute_alpha(n, S_d, d_min):
        """Approximate the alpha of a power law distribution.

        Parameters
        ----------
        n: int or np.array of int
            Number of entries that are larger than or equal to d_min
        S_d: float or np.array of float
            Sum of log degrees in the distribution that are larger than or equal to d_min
        d_min: int
            The minimum degree of nodes to consider

        Returns
        -------
        alpha: float
            The estimated alpha of the power law distribution
        """

        return n / (S_d - n * torch.log(d_min - 0.5)) + 1

    @staticmethod
    def update_Sx(S_old, n_old, d_old, d_new, d_min):
        """Update on the sum of log degrees S_d and n based on degree distribution resulting from inserting or deleting
        a single edges.

        Parameters
        ----------
        S_old: float
            Sum of log degrees in the distribution that are larger than or equal to d_min.
        n_old: int
            Number of entries in the old distribution that are larger than or equal to d_min.
        d_old: np.array, shape [num_nodes,] dtype int
            The old degree sequence.
        d_new: np.array, shape [num_nodes,] dtype int
            The new degree sequence
        d_min: int
            The minimum degree of nodes to consider

        Returns
        -------
        new_S_d: float, the updated sum of log degrees in the distribution that are larger than or equal to d_min.
        new_n: int, the updated number of entries in the old distribution that are larger than or equal to d_min.
        """

        old_in_range = d_old >= d_min
        new_in_range = d_new >= d_min

        d_old_in_range = torch.mul(d_old, old_in_range)
        d_new_in_range = torch.mul(d_new, new_in_range)

        new_S_d = S_old - torch.log(torch.maximum(d_old_in_range, 1)).sum(1) + torch.log(
            torch.maximum(d_new_in_range, 1)).sum(1)
        new_n = n_old - torch.sum(old_in_range, 1) + torch.sum(new_in_range, 1)

        return new_S_d, new_n

    @staticmethod
    def compute_log_likelihood(n, alpha, S_d, d_min):
        """Compute log likelihood of the powerlaw fit.

        Parameters
        ----------
        n: int
            Number of entries in the old distribution that are larger than or equal to d_min.
        alpha: float
            The estimated alpha of the power law distribution
        S_d: float
            Sum of log degrees in the distribution that are larger than or equal to d_min.
        d_min: int
            The minimum degree of nodes to consider

        Returns
        -------
        float: the estimated log likelihood
        """

        return n * torch.log(alpha) + n * alpha * torch.log(d_min) + (alpha + 1) * S_d

    @staticmethod
    def filter_chisquare(ll_ratios: Tensor, cutoff: float):
        return ll_ratios < cutoff
