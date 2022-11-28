import torch
from torch import Tensor


def project(num_budgets: int, values: Tensor, eps: float = 1e-7) -> Tensor:
    r"""Project :obj:`values`:
    :math:`num_budgets \ge \sum \Pi_{[0, 1]}(\text{values})`."""
    if torch.clamp(values, 0, 1).sum() > num_budgets:
        left = (values - 1).min()
        right = values.max()
        miu = bisection(values, left, right, num_budgets)
        values = values - miu
    return torch.clamp(values, min=eps, max=1 - eps)


def bisection(edge_weight: Tensor, a: float, b: float, n_pert: int, eps=1e-5,
              max_iter=1e3) -> Tensor:
    """Bisection search for projection."""
    def shift(offset: float):
        return (torch.clamp(edge_weight - offset, 0, 1).sum() - n_pert)

    miu = a
    for _ in range(int(max_iter)):
        miu = (a + b) / 2
        # Check if middle point is root
        if (shift(miu) == 0.0):
            break
        # Decide the side to repeat the steps
        if (shift(miu) * shift(a) < 0):
            b = miu
        else:
            a = miu
        if ((b - a) <= eps):
            break
    return miu


def num_possible_edges(n: int, is_undirected_graph: bool) -> int:
    """Determine number of possible edges for graph."""
    if is_undirected_graph:
        return n * (n - 1) // 2
    else:
        return int(n**2)  # We filter self-loops later


def linear_to_triu_idx(n: int, lin_idx: Tensor) -> Tensor:
    """Linear index to upper triangular matrix without diagonal.
    This is similar to
    https://stackoverflow.com/questions/242711/algorithm-for-index-numbers-of-triangular-matrix-coefficients/28116498#28116498
    with number nodes decremented and col index incremented by one."""
    nn = n * (n - 1)
    row_idx = n - 2 - torch.floor(
        torch.sqrt(-8 * lin_idx.double() + 4 * nn - 7) / 2.0 - 0.5).long()
    col_idx = 1 + lin_idx + row_idx - nn // 2 + torch.div(
        (n - row_idx) * (n - row_idx - 1), 2, rounding_mode='floor')
    return torch.stack((row_idx, col_idx))


def linear_to_full_idx(n: int, lin_idx: Tensor) -> Tensor:
    """Linear index to dense matrix including diagonal."""
    row_idx = torch.div(lin_idx, n, rounding_mode='floor')
    col_idx = lin_idx % n
    return torch.stack((row_idx, col_idx))
