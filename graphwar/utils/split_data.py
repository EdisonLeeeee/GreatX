import torch
from torch import Tensor
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from graphwar.utils import BunchDict


def train_val_test_split_tabular(N: int, *,
                                 train: float = 0.1,
                                 test: float = 0.8,
                                 val: float = 0.1,
                                 stratify=None,
                                 random_state: Optional[int] = None) -> Tuple:

    idx = torch.arange(N)
    idx_train, idx_test = train_test_split(idx,
                                           random_state=random_state,
                                           train_size=train + val,
                                           test_size=test,
                                           stratify=stratify)
    if val:
        stratify = stratify[idx_train]
        idx_train, idx_val = train_test_split(idx_train,
                                              random_state=random_state,
                                              train_size=train / (train + val),
                                              stratify=stratify)
    else:
        idx_val = None

    return idx_train, idx_val, idx_test


def split_nodes(labels: Tensor, *,
                train: float = 0.1,
                test: float = 0.8,
                val: float = 0.1,
                random_state: Optional[int] = None) -> BunchDict:

    val = 0. if val is None else val
    assert train + val + test <= 1.0

    train_nodes, val_nodes, test_nodes = train_val_test_split_tabular(
        labels.shape[0],
        train=train,
        val=val,
        test=test,
        stratify=labels,
        random_state=random_state)

    return BunchDict(
        dict(train_nodes=train_nodes,
             val_nodes=val_nodes,
             test_nodes=test_nodes))
