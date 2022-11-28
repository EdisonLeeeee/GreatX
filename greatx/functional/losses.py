from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def margin_loss(score: Tensor, labels: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
    r"""Margin loss between true score and highest non-target score:

    .. math::
        m = - s_{y} + max_{y' \ne y} s_{y'}

    where :math:`m` is the margin :math:`s` the score and :math:`y` the
    labels.

    Args:
        score (Tensor): Some score (e.g. logits) of shape
            :obj:`[n_elem, dim]`.
        labels (LongTensor): The labels of shape :obj:`[n_elem]`.
        mask (Tensor, optional): To select subset of `score` and
            `labels` of shape :obj:`[n_select]`. Defaults to None.

    :rtype: (Tensor)
    """
    if mask is not None:
        score = score[mask]
        labels = labels[mask]

    linear_idx = torch.arange(score.size(0), device=score.device)
    true_score = score[linear_idx, labels]

    score = score.clone()
    score[linear_idx, labels] = float('-Inf')
    best_non_target_score = score.amax(dim=-1)

    margin = best_non_target_score - true_score
    return margin


def tanh_margin_loss(prediction: Tensor, labels: Tensor,
                     mask: Optional[Tensor] = None) -> Tensor:
    """Calculate tanh margin loss, a node-classification loss that focuses
    on nodes next to decision boundary.

    Args:
        prediction (Tensor): Prediction of shape :obj:`[n_elem, dim]`.
        labels (LongTensor): The labels of shape :obj:`[n_elem]`.
        mask (Tensor, optional): To select subset of `score` and
            `labels` of shape :obj:`[n_select]`. Defaults to None.

    :rtype: (Tensor)
    """
    log_logits = F.log_softmax(prediction, dim=-1)
    margin = margin_loss(log_logits, labels, mask)
    loss = torch.tanh(margin).mean()
    return loss


def probability_margin_loss(prediction: Tensor, labels: Tensor,
                            mask: Optional[Tensor] = None) -> Tensor:
    """Calculate probability margin loss, a node-classification loss that
    focuses  on nodes next to decision boundary. See `Are Defenses for
    Graph Neural Networks Robust?
    <https://www.cs.cit.tum.de/daml/are-gnn-defenses-robust>`_ for details.

    Args:
        prediction (Tensor): Prediction of shape :obj:`[n_elem, dim]`.
        labels (LongTensor): The labels of shape :obj:`[n_elem]`.
        mask (Tensor, optional): To select subset of `score` and
            `labels` of shape :obj:`[n_select]`. Defaults to None.

    :rtype: (Tensor)
    """
    logits = F.softmax(prediction, dim=-1)
    margin = margin_loss(logits, labels, mask)
    return margin.mean()


def masked_cross_entropy(log_logits: Tensor, labels: Tensor,
                         mask: Optional[Tensor] = None) -> Tensor:
    """Calculate masked cross entropy loss, a node-classification loss that
    focuses on nodes next to decision boundary.

    Args:
        log_logits (Tensor): Log logits of shape :obj:`[n_elem, dim]`.
        labels (LongTensor): The labels of shape :obj:`[n_elem]`.
        mask (Tensor, optional): To select subset of `score` and
            `labels` of shape :obj:`[n_select]`. Defaults to None.

    :rtype: (Tensor)
    """
    if mask is not None:
        log_logits = log_logits[mask]
        labels = labels[mask]

    is_correct = log_logits.argmax(-1) == labels
    if is_correct.any():
        log_logits = log_logits[is_correct]
        labels = labels[is_correct]

    loss = F.cross_entropy(log_logits, labels)
    return loss
