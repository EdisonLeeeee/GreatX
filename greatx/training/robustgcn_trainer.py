from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from greatx.training.trainer import Trainer


class RobustGCNTrainer(Trainer):
    """Custom trainer for :class:`~greatx.nn.models.supervised.RobustGCN`

    Parameters
    ----------
    model : nn.Module
        the model used for training
    device : Union[str, torch.device], optional
        the device used for training, by default 'cpu'
    cfg : other keyword arguments, such as `lr` and `weight_decay`.

    Note
    ----
    :class:`~greatx.training.RobustGCNTrainer` accepts the following argument:

    * :obj:`kl`: trade-off parameter for kl loss

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

        if mask is not None:
            out = out[mask]
            y = y[mask]

        # ================= add KL loss here =============================
        kl = self.cfg.get('kl', 5e-4)
        mean, var = model.mean, model.var
        kl_loss = -0.5 * torch.sum(
            torch.mean(1 + torch.log(var + 1e-8) - mean.pow(2) + var, dim=1))
        loss = F.cross_entropy(out, y) + kl * kl_loss
        # ===============================================================
        loss.backward()
        self.callbacks.on_train_batch_end(0)
        return dict(loss=loss.item(),
                    acc=out.argmax(1).eq(y).float().mean().item())
