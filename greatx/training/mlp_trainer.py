import torch
import torch.nn.functional as F
from torch import Tensor
from greatx.training import Trainer


class MLPTrainer(Trainer):
    """Simple trainer for input-to-output training.

    Parameters
    ----------
    model : nn.Module
        the model used for training
    device : Union[str, torch.device], optional
        the device used for training, by default 'cpu'
    cfg : other keyword arguments, such as `lr` and `weight_decay`.    

    Example
    -------
    >>> from greatx.training import MLPTrainer
    >>> model = ... # your model
    >>> trainer = MLPTrainer(model, device='cuda')

    >>> data # PyG-like data, e.g., Cora
    Data(x=[2485, 1433], edge_index=[2, 10138], y=[2485])

    >>> # simple training
    >>> trainer.fit({'x': data.x, 'y': data.y, 'mask': your_train_mask})

    >>> # train with model picking
    >>> from greatx.training import ModelCheckpoint
    >>> cb = ModelCheckpoint('my_ckpt', monitor='val_acc')
    >>> trainer.fit({'x': data.x, 'y': data.y, 'mask': your_train_mask}, 
    ... {'x': data.x, 'y': data.y, 'mask': your_val_mask}, callbacks=[cb])    

    >>> # get training logs
    >>> history = trainer.model.history

    >>> trainer.evaluate({'x': data.x, 'y': data.y, 'mask': your_test_mask}) # evaluation

    >>> predict = trainer.predict({'x': data.x, 'y': data.y, 'mask': your_mask}) # prediction
    """

    def train_step(self, inputs: dict) -> dict:
        """One-step training on the inputs.

        Parameters
        ----------
        inputs : dict like or custom inputs
            the training data.

        Returns
        -------
        dict
            the output logs, including `loss` and `acc`, etc.
        """
        model = self.model
        self.callbacks.on_train_batch_begin(0)

        model.train()
        x = inputs['x'].to(self.device)
        y = inputs['y'].squeeze().to(self.device)
        mask = inputs.get('mask', None)

        out = model(x)

        if mask is not None:
            out = out[mask]
            y = y[mask]

        loss = F.cross_entropy(out, y)
        loss.backward()
        self.callbacks.on_train_batch_end(0)

        return dict(loss=loss.item(), acc=out.argmax(1).eq(y).float().mean().item())

    @torch.no_grad()
    def test_step(self, inputs: dict) -> dict:
        """One-step evaluation on the inputs.

        Parameters
        ----------
        inputs : dict like or custom inputs
            the testing data.

        Returns
        -------
        dict
            the output logs, including `loss` and `acc`, etc.
        """
        model = self.model
        model.eval()
        x = inputs['x'].to(self.device)
        y = inputs['y'].squeeze().to(self.device)
        mask = inputs.get('mask', None)

        out = model(x)

        if mask is not None:
            out = out[mask]
            y = y[mask]

        loss = F.cross_entropy(out, y)

        return dict(loss=loss.item(), acc=out.argmax(1).eq(y).float().mean().item())

    @torch.no_grad()
    def predict_step(self, inputs: dict) -> Tensor:
        """One-step prediction on the inputs.

        Parameters
        ----------
        inputs : dict like or custom inputs
            the prediction data.

        Returns
        -------
        Tensor
            the output prediction.
        """
        model = self.model
        model.eval()
        callbacks = self.callbacks
        x = inputs['x'].to(self.device)
        mask = inputs.get('mask', None)

        out = model(x)

        if mask is not None:
            out = out[mask]
        return out
