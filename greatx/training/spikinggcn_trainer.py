from typing import Optional

import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from greatx.training.trainer import Trainer


class SpikingGCNTrainer(Trainer):
    """Custom trainer for :class:`~greatx.nn.models.supervised.SpikingGCN`

    Parameters
    ----------
    model : nn.Module
        the model used for training
    device : Union[str, torch.device], optional
        the device used for training, by default 'cpu'
    cfg : other keyword arguments, such as `lr` and `weight_decay`.

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
            out = model(data.x, data.edge_index, data.edge_weight)
        else:
            out = model(data.x, adj_t)

        y_one_hot = F.one_hot(y, out.size(-1)).float()

        if mask is not None:
            out = out[mask]
            y = y[mask]
            y_one_hot = y_one_hot[mask]

        # ================= MSE loss here ====================
        loss = F.mse_loss(out, y_one_hot)
        # =====================================================

        loss.backward()
        self.callbacks.on_train_batch_end(0)
        return dict(loss=loss.item(),
                    acc=out.argmax(1).eq(y).float().mean().item())
