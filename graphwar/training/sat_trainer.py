import torch
import torch.nn.functional as F
from torch import Tensor
from graphwar.training import Trainer


class SATTrainer(Trainer):
    """Custom trainer for :class:`graphwar.nn.models.SAT`

    Parameters
    ----------
    model : nn.Module
        the model used for training
    device : Union[str, torch.device], optional
        the device used for training, by default 'cpu'
    cfg : other keyword arguments, such as `lr` and `weight_decay`.   

    Note
    ----
    :class:`graphwar.training.SATTrainer` accepts the following additional arguments:   

    * :obj:`eps_U`: scale of perturbation on eigenvectors
    * :obj:`eps_V`: scale of perturbation on eigenvalues
    * :obj:`lambda_U`: trade-off parameters for eigenvectors-specific loss
    * :obj:`lambda_V`: trade-off parameters for eigenvalues-specific loss

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

        eps_U = self.cfg.get("eps_U", 0.1)
        eps_V = self.cfg.get("eps_V", 0.1)
        lamb_U = self.cfg.get("lamb_U", 0.5)
        lamb_V = self.cfg.get("lamb_V", 0.5)

        data = inputs['data'].to(self.device)
        mask = inputs.get('mask', None)
        U, V = data.U, data.V
        y = data.y

        U.requires_grad_()
        V.requires_grad_()

        out = model(data.x, U, V)

        if mask is not None:
            out = out[mask]
            y = y[mask]

        loss = F.cross_entropy(out, y)
        U_grad, V_grad = torch.autograd.grad(loss, [U, V], retain_graph=True)

        U.requires_grad_(False)
        V.requires_grad_(False)

        U_grad = eps_U * U_grad / torch.norm(U_grad, 2)
        V_grad = eps_V * V_grad / torch.norm(V_grad, 2)

        out_U = model(data.x, U + U_grad, V)
        out_V = model(data.x, U, V + V_grad)

        if mask is not None:
            out_U = out_U[mask]
            out_V = out_V[mask]

        loss += lamb_U * \
            F.cross_entropy(out_U, y) + lamb_V * F.cross_entropy(out_V, y)
        # ===============================================================

        loss.backward()
        self.callbacks.on_train_batch_end(0)
        return dict(loss=loss.item(), acc=out.argmax(1).eq(y).float().mean().item())
