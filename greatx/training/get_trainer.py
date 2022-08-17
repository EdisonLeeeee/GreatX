from typing import Union

import torch

from greatx import training


def get_trainer(model: Union[str, torch.nn.Module]) -> training.trainer.Trainer:
    """Get the default trainer using str or a model in :class:`~greatx.nn.models.supervised`

    Parameters
    ----------
    model : Union[str, torch.nn.Module]
        the model to be trained in

    Returns
    -------
    Custom trainer or default trainer :class:`~greatx.training.Trainer` for the model.

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
    """
    default = training.Trainer
    if isinstance(model, str):
        class_name = model
    else:
        class_name = model.__class__.__name__

    trainer = getattr(training, class_name + "Trainer", default)
    return trainer
