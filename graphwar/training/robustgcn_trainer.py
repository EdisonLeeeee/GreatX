import torch
import torch.nn.functional as F

from graphwar.training import Trainer


class RobustGCNTrainer(Trainer):
    """Custom trainer for :class:`graphwar.nn.models.RobustGCN`

    Parameters
    ----------
    model : nn.Module
        the model used for training
    device : Union[str, torch.device], optional
        the device used for training, by default 'cpu'
    cfg : other keyword arguments, such as `lr` and `weight_decay`.      

    Note
    ----
    :class:`graphwar.training.RobustGCNTrainer` accepts the following additional arguments: 

    * :obj:`kl`: trade-off parameter for kl loss      

    """

    def train_step(self, inputs: dict) -> dict:
        """One-step training on the input dataloader.

        Parameters
        ----------
        inputs : dict
            the training data.

        Returns
        -------
        dict
            the output logs, including `loss` and `val_acc`, etc.
        """
        model = self.model
        self.callbacks.on_train_batch_begin(0)

        model.train()
        data = inputs['data'].to(self.device)
        mask = inputs.get('mask', None)
        adj_t = getattr(data, 'adj_t', None)
        y = data.y

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
        kl_loss = -0.5 * torch.sum(torch.mean(1 + torch.log(var + 1e-8) -
                                              mean.pow(2) + var, dim=1))
        loss = F.cross_entropy(out, y) + kl * kl_loss
        # ===============================================================
        loss.backward()
        self.callbacks.on_train_batch_end(0)
        return dict(loss=loss.item(), acc=out.argmax(1).eq(y).float().mean().item())
