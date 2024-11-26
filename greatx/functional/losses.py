import torch
import torch.nn.functional as F
from torch import Tensor


def margin_loss(score: Tensor, target: Tensor) -> Tensor:
    r"""Margin loss between true score and highest non-target score:

    .. math::
        m = - s_{y} + max_{y' \ne y} s_{y'}

    where :math:`m` is the margin :math:`s` the score and :math:`y` the
    target.

    Parameters
    ----------
    score : Tensor
        some score (e.g. prediction) of shape :obj:`[n_elem, dim]`.
    target : LongTensor
        the target of shape :obj:`[n_elem]`.

    Returns
    -------
    Tensor
        the calculated margins
    """

    linear_idx = torch.arange(score.size(0), device=score.device)
    true_score = score[linear_idx, target]

    score = score.clone()
    score[linear_idx, target] = float('-Inf')
    best_non_target_score = score.amax(dim=-1)

    margin = best_non_target_score - true_score
    return margin


def tanh_margin_loss(prediction: Tensor, target: Tensor) -> Tensor:
    r"""Calculate tanh margin loss, a node-classification loss that focuses
    on nodes next to decision boundary.

    Parameters
    ----------
    prediction : Tensor
        prediction of shape :obj:`[n_elem, dim]`.
    target : LongTensor
        the target of shape :obj:`[n_elem]`.

    Returns
    -------
    Tensor
        the calculated loss
    """
    prediction = F.log_softmax(prediction, dim=-1)
    margin = margin_loss(prediction, target)
    loss = torch.tanh(margin).mean()
    return loss


def probability_margin_loss(prediction: Tensor, target: Tensor) -> Tensor:
    r"""Calculate probability margin loss, a node-classification loss that
    focuses  on nodes next to decision boundary. See `Are Defenses for
    Graph Neural Networks Robust?
    <https://www.cs.cit.tum.de/daml/are-gnn-defenses-robust>`_ for details.

    Parameters
    ----------
    prediction : Tensor
        prediction of shape :obj:`[n_elem, dim]`.
    target : LongTensor
        the target of shape :obj:`[n_elem]`.

    Returns
    -------
    Tensor
        the calculated loss
    """
    prediction = F.softmax(prediction, dim=-1)
    margin = margin_loss(prediction, target)
    return margin.mean()


def masked_cross_entropy(prediction: Tensor, target: Tensor) -> Tensor:
    r"""Calculate masked cross entropy loss, a node-classification loss that
    focuses on nodes next to decision boundary.

    Parameters
    ----------
    prediction : Tensor
        prediction of shape :obj:`[n_elem, dim]`.
    target : LongTensor
        the target of shape :obj:`[n_elem]`.

    Returns
    -------
    Tensor
        the calculated loss
    """

    is_correct = prediction.argmax(-1) == target
    if is_correct.any():
        prediction = prediction[is_correct]
        target = target[is_correct]

    loss = F.cross_entropy(prediction, target)
    return loss
