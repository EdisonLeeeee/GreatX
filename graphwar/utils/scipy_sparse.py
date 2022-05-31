import scipy.sparse as sp
import numpy as np


def scipy_normalize(adj_matrix: sp.csr_matrix, add_self_loops: bool = True):
    if add_self_loops:
        adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0],
                                         dtype=adj_matrix.dtype,
                                         format='csr')
    degree = np.maximum(adj_matrix.sum(1).A1, 1)
    norm = sp.diags(np.power(degree, -0.5))
    adj_matrix = norm @ adj_matrix @ norm
    return adj_matrix
