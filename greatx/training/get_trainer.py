from typing import Union

import torch

from greatx import training

Trainer = training.trainer.Trainer
UnspuervisedTrainer = training.unsup_trainer.UnspuervisedTrainer


def get_trainer(model: Union[str, torch.nn.Module]) -> Trainer:
    r"""Get the default trainer using str or a model in
    :class:`greatx.nn.models`

    Parameters
    ----------
    model : Union[str, torch.nn.Module]
        the model to be trained

    Returns
    -------
    Custom trainer or default trainer
    :class:`greatx.training.Trainer` for the model.

    Examples
    --------
    >>> import greatx
    >>> greatx.training.get_trainer('GCN')
    greatx.training.trainer.Trainer

    >>> from greatx.nn.models import GCN
    >>> greatx.training.get_trainer(GCN)
    greatx.training.trainer.Trainer

    >>> # by default, it returns `greatx.training.Trainer`
    >>> greatx.training.get_trainer('unimplemeted_model')
    greatx.training.trainer.Trainer

    >>> greatx.training.get_trainer('RobustGCN')
    greatx.training.robustgcn_trainer.RobustGCNTrainer

    >>> # it is case-sensitive
    >>> greatx.training.get_trainer('robustGCN')
    greatx.training.trainer.Trainer

    Note
    ----
    Unsupervised models are not supported for thie method to get
    the :class:`greatx.training.UnsupervisedTrainer`. It will also
    return the :class:`greatx.training.Trainer` by default.
    """
    default = training.Trainer
    if isinstance(model, str):
        class_name = model
    else:
        class_name = model.__class__.__name__

    trainer = getattr(training, class_name + "Trainer", default)
    return trainer
