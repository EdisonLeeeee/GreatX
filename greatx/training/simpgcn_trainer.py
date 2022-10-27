from typing import Optional

import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from greatx.training.trainer import Trainer


class SimPGCNTrainer(Trainer):
    """Custom trainer for :class:`~greatx.nn.models.supervised.SimPGCN`

    Parameters
    ----------
    model : nn.Module
        the model used for training
    device : Union[str, torch.device], optional
        the device used for training, by default 'cpu'
    cfg : other keyword arguments, such as `lr` and `weight_decay`.

    Note
    ----
    :class:`~greatx.training.SimPGCNTrainer` accepts the
    following additional argument:

    * :obj:`lambda_`: trade-off parameter for regression loss

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
        y = data.y.squeeze()

        if adj_t is None:
            out, embeddings = model(data.x, data.edge_index, data.edge_weight)
        else:
            out, embeddings = model(data.x, adj_t)

        if mask is not None:
            out = out[mask]
            y = y[mask]

        # ================= add regression loss here ====================
        lambda_ = self.cfg.get("lambda_", 5.0)
        loss = F.cross_entropy(out, y) + lambda_ * \
            model.regression_loss(embeddings)
        # ===============================================================
        loss.backward()
        self.callbacks.on_train_batch_end(0)
        return dict(loss=loss.item(),
                    acc=out.argmax(1).eq(y).float().mean().item())
