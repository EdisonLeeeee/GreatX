from typing import Optional, Union

import numpy as np
import torch


def mark(
    logit: Union[np.ndarray, torch.Tensor],
    y_true: Optional[int] = None,
) -> str:
    """Marks the predicted classes and corresponding
    probability for a better view.

    Parameters
    ----------
    logit : Union[np.ndarray, torch.Tensor]
        the predicted class probability
    y_true: Optional[int]
        the ground truth label, by default None

    Returns
    -------
    str
        the formated string

    Examples
    --------
    >>> from greatx.utils import mark
    >>> import torch
    >>> pred = torch.tensor([0.5, 0.3, 0.1, 0.1])
    >>> print(mark(pred))
    Prediction (y=0): 0.500, 0.300, 0.100, 0.100
                       👆(0)
    >>> print(mark(pred, y_true=2))
    Ground Truth (y=2):                👇(2)
    Prediction (pred=0): 0.500, 0.300, 0.100, 0.100
                          👆(0)
    Margin: -0.400
    """
    assert (isinstance(logit, (np.ndarray, torch.Tensor)) and logit.ndim == 1)

    y_pred = logit.argmax()
    num_classes = len(logit)
    string = ""
    if y_true is not None:
        assert 0 <= y_true < num_classes
        string += f"Ground Truth (y={y_true}): "
        string += 7 * ' ' * y_true + f' 👇({y_true}) '
        string += 7 * ' ' * (num_classes - y_true)
        string += "\n"
    string += f"Prediction (pred={y_pred}): "
    string += ', '.join([f"{x:.3f}" for x in logit])
    string += '\n' + ' ' * 20
    string += 7 * ' ' * y_pred + f' 👆({y_pred}) '
    string += 7 * ' ' * (num_classes - y_pred)
    if y_true is not None:
        string += f"\nMargin: {logit[y_true]-logit[y_pred]:.3f}"

    return string
