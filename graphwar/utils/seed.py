import random
from numbers import Number
from typing import Optional

import numpy as np
import torch

__all__ = ["set_seed"]


def set_seed(seed: Optional[int] = None) -> Optional[int]:
    """Set random seed for reproduction.

    Parameters
    ----------
    seed : Optional[int], optional
        random seed, by default None

    Returns
    -------
    Optional[int]
        the random seed.

    Example
    -------
    >>> from graphwar import set_seed
    >>> set_seed(42)
    42

    """
    assert seed is None or isinstance(seed, Number), seed
    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
    return seed
