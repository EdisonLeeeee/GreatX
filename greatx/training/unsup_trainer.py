from typing import Optional

from torch import Tensor
from torch_geometric.data import Data

from greatx.training import Trainer


class UnspuervisedTrainer(Trainer):
    r"""Custom trainer for Unspuervised models.

    Parameters
    ----------
    model : nn.Module
        the model used for training
    device : Union[str, torch.device], optional
        the device used for training, by default 'cpu'
    cfg : other keyword arguments, such as `lr` and `weight_decay`.

    Note
    ----
    The method :meth:`loss` must be imppleted in :obj:`model`.
    """
    def train_step(self, data: Data, mask: Optional[Tensor] = None) -> dict:
        """One-step training on the inputs.

        Parameters
        ----------
        data : Data
            the training data.
        mask : Optional[Tensor]
            the mask of training nodes.

        Returns
        -------
        dict
            the output logs, including `loss` and `acc`, etc.
        """
        model = self.model
        self.callbacks.on_train_batch_begin(0)

        model.train()
        data = data.to(self.device)
        adj_t = getattr(data, 'adj_t', None)

        if adj_t is None:
            out = model(data.x, data.edge_index, data.edge_weight)
        else:
            out = model(data.x, adj_t)

        if not isinstance(out, tuple):
            out = out,

        loss = model.loss(*out)

        loss.backward()
        self.callbacks.on_train_batch_end(0)
        return dict(loss=loss.item())

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError
