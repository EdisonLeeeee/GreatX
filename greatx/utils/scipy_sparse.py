import scipy.sparse as sp
import numpy as np


def scipy_normalize(adj_matrix: sp.csr_matrix,
                    add_self_loops: bool = True) -> sp.csr_matrix:
    """Normalize a sparse matrix according to :obj:`GCN`
    from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper (ICLR'17)

    Parameters
    ----------
    adj_matrix : sp.csr_matrix
        the input sparse matrix denoting a graph.
    add_self_loops : bool, optional
        whether to add self-loops, by default True

    Returns
    -------
    sp.csr_matrix
        the normalized adjacency matrix.
    """
    if add_self_loops:
        adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0],
                                         dtype=adj_matrix.dtype,
                                         format='csr')
    degree = np.maximum(adj_matrix.sum(1).A1, 1)
    norm = sp.diags(np.power(degree, -0.5))
    adj_matrix = norm @ adj_matrix @ norm
    return adj_matrix
