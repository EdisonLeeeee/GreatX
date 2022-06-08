from typing import Union

import torch

from graphwar import training


def get_trainer(model: Union[str, torch.nn.Module]) -> training.trainer.Trainer:
    """Get the default trainer using str or a model in :class:`graphwar.nn.models`

    Parameters
    ----------
    model : Union[str, torch.nn.Module]
        the model to be trained in

    Returns
    -------
    Custom trainer or default trainer :class:`graphwar.training.Trainer` for the model.

    Examples
    --------
    >>> import graphwar
    >>> graphwar.training.get_trainer('GCN')
    graphwar.training.trainer.Trainer

    >>> from graphwar.nn.models import GCN
    >>> graphwar.training.get_trainer(GCN)
    graphwar.training.trainer.Trainer

    >>> # by default, it returns `graphwar.training.Trainer`
    >>> graphwar.training.get_trainer('unimplemeted_model')
    graphwar.training.trainer.Trainer

    >>> graphwar.training.get_trainer('RobustGCN')
    graphwar.training.robustgcn_trainer.RobustGCNTrainer

    >>> # it is case-sensitive
    >>> graphwar.training.get_trainer('robustGCN')
    graphwar.training.trainer.Trainer
    """
    default = training.Trainer
    if isinstance(model, str):
        class_name = model
    else:
        class_name = model.__class__.__name__

    trainer = getattr(training, class_name + "Trainer", default)
    return trainer
