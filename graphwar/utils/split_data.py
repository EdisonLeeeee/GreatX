from typing import Optional, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch import Tensor

from graphwar.utils import BunchDict


def split_nodes(labels: Tensor, *,
                train: float = 0.1,
                test: float = 0.8,
                val: float = 0.1,
                random_state: Optional[int] = None) -> BunchDict:
    """Randomly split a set of nodes labeled with :obj:`labels`.

    Parameters
    ----------
    labels : Tensor
        the labels of the nodes.
    train : float, optional
        the percentage of the training set, by default 0.1
    test : float, optional
        the percentage of the test set, by default 0.8
    val : float, optional
        the percentage of the validation set, by default 0.1
    random_state : Optional[int], optional
        random seed for the random number generator, by default None

    Returns
    -------
    BunchDict with the following items:
        * train_nodes: torch.Tensor with Size [train * num_nodes]
            The indices of the training nodes
        * val_nodes: torch.Tensor with Size [val * num_nodes]
            The indices of the validation nodes
        * test_nodes torch.Tensor with Size [test * num_nodes]
            The indices of the test nodes        
    """

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


def split_nodes_by_classes(labels: torch.Tensor,
                           n_per_class: int = 20,
                           random_state: Optional[int] = None) -> BunchDict:
    """Randomly split the training data by the number of nodes per classes.

    Parameters
    ----------
    labels: torch.Tensor [num_nodes]
        The class labels
    n_per_class : int
        Number of samples per class
    random_state: Optional[int]
        Random seed

    Returns
    -------
    BunchDict with the following items:
        * train_nodes: torch.Tensor with Size [n_per_class * num_classes]
            The indices of the training nodes
        * val_nodes: torch.Tensor with Size [n_per_class * num_classes]
            The indices of the validation nodes
        * test_nodes torch.Tensor with Size [num_nodes - 2*n_per_class * num_classes]
            The indices of the test nodes
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    num_classes = labels.max() + 1

    split_train, split_val = [], []
    for c in range(num_classes):
        perm = (labels == c).nonzero().view(-1)
        perm = perm[torch.randperm(perm.size(0))]
        split_train.append(perm[:n_per_class])
        split_val.append(perm[n_per_class:2 * n_per_class])

    split_train = torch.cat(split_train)
    split_train = split_train[torch.randperm(split_train.size(0))]
    split_val = torch.cat(split_val)
    split_train = split_val[torch.randperm(split_val.size(0))]

    assert split_train.size(0) == split_val.size(
        0) == n_per_class * num_classes

    mask = torch.ones_like(labels).bool()

    mask[split_train] = False
    mask[split_val] = False
    split_test = torch.arange(labels.size(0), device=labels.device)[mask]

    return BunchDict(
        dict(train_nodes=split_train,
             val_nodes=split_val,
             test_nodes=split_test))


def train_val_test_split_tabular(N: int, *,
                                 train: float = 0.1,
                                 test: float = 0.8,
                                 val: float = 0.1,
                                 stratify: Optional[bool] = None,
                                 random_state: Optional[int] = None) -> Tuple:

    idx = torch.arange(N)
    idx_train, idx_test = train_test_split(idx,
                                           random_state=random_state,
                                           train_size=train + val,
                                           test_size=test,
                                           stratify=stratify)
    if val:
        if stratify is not None:
            stratify = stratify[idx_train]
        idx_train, idx_val = train_test_split(idx_train,
                                              random_state=random_state,
                                              train_size=train / (train + val),
                                              stratify=stratify)
    else:
        idx_val = None

    return idx_train, idx_val, idx_test
