import torch.nn.functional as F
from graphwar.training import Trainer


class SimPGCNTrainer(Trainer):
    """Custom trainer for :class:`graphwar.nn.models.SimPGCN`

    Parameters
    ----------
    model : nn.Module
        the model used for training
    device : Union[str, torch.device], optional
        the device used for training, by default 'cpu'
    cfg : other keyword arguments, such as `lr` and `weight_decay`.    

    Note
    ----
    :class:`graphwar.training.SimPGCNTrainer` accepts the following additional arguments:   

    * :obj:`lambda_`: trade-off parameter for regression loss

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
        return dict(loss=loss.item(), acc=out.argmax(1).eq(y).float().mean().item())
