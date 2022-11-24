from greatx.training import Trainer


class UnspuervisedTrainer(Trainer):
    r"""Custom trainer for Unspuervised models, similar to
    :class:`greatx.training.Trainer` but only uses unsupervised
    loss defined in :meth:`model.loss()` method.

    See also
    --------
    :class:`greatx.training.Trainer`
    """

    supervised = False

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError
