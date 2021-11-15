import sys
import torch
import torch.nn as nn

from torch import Tensor
from dgl import DGLGraph
from typing import Optional, Union, Any, Callable, List
from torch.utils.data import DataLoader

from graphwar.metrics import Accuracy
from graphwar.training.callbacks import Callback, CallbackList
from graphwar.utils import BunchDict, Progbar
from graphwar import Info

_FEATURE = Info.feat


class Trainer:
    """A simple trainer to train graph neural network models conveniently.


    Example
    -------
    >>> from graphwar.trainig import Trainer
    >>> model = ... # your model
    >>> trainer = Trainer(model, device='cuda')

    >>> # simple training
    >>> g = ... # DGL graph
    >>> g.ndata['feat'] = feat # setup node features
    >>> y_train = ... # trainig node labels
    >>> train_nodes = ... #  training nodes
    >>> trainer.fit(g, y=y_train, index=train_nodes)

    >>> # train with model picking
    >>> g = ... # DGL graph
    >>> g.ndata['feat'] = feat # setup node features
    >>> y_train = ... # trainig node labels
    >>> train_nodes = ... #  training nodes
    >>> y_val = ... # validation node labels
    >>> val_nodes = ... #  validation nodes    
    >>> from graphwar.trainig import ModelCheckpoint
    >>> cb = ModelCheckpoint('my_ckpy', monitor='val_accuracy)
    >>> trainer.fit(g, y=y_train, index=train_nodes, val_y=y_val, val_index=val_nodes, callbacks=[cb])    

    >>> # get training logs
    >>> history = trainer.model.history

    >>> y_test = ... # testing node labels
    >>> test_nodes = ... # testing nodes
    >>> trainer.evaluate(g, y=y_test, index=test_nodes)

    >>> trainer.predict(g, index=y_test)

    >>> import torch.nn as nn
    >>> trainer.predict(g, index=y_test, transform=nn.Softmax(dim=-1))

    """

    def __init__(self, model: nn.Module, device: Union[str, torch.device] = 'cpu', **cfg):
        """Initialize a Trainer object.

        Parameters
        ----------
        model : nn.Module
            the model used for training
        device : Union[str, torch.device], optional
            the device used for training, by default 'cpu'
        cfg : other keyword arguments, such as `lr` and `weight_decay`.
        """
        self.device = torch.device(device)
        self.model = model.to(device)
        self.cfg = BunchDict(cfg)
        self.optimizer = self.config_optimizer()
        self.loss = self.config_loss()
        metrics = self.config_metrics()
        if not isinstance(metrics, list):
            metrics = [metrics]
        self.metrics = metrics

    def fit(self, g: DGLGraph, y: Optional[Tensor] = None, index: Optional[Tensor] = None,
            val_g: Optional[DGLGraph] = None, val_y: Optional[Tensor] = None, val_index: Optional[Tensor] = None,
            callbacks: Optional[Callback] = None, verbose: Optional[int] = 1, epochs: int = 100) -> "Trainer":
        """Simple training method designed for `:attr:model`

        Parameters
        ----------
        g : DGLGraph
            the dgl graph used for training
        y : Optional[Tensor], optional
            the training labels, by default None
        index : Optional[Tensor], optional
            the training index/mask, such as training nodes 
            index or mask, by default None
        val_g : Optional[DGLGraph], optional
            the dgl graph used for validation, if None, 
            it will set as `g`, by default None
        val_y : Optional[Tensor], optional
            the validation labels, by default None
        val_index : Optional[Tensor], optional
            the validation index/mask, such as validation nodes 
            index or mask, by default None
        callbacks : Optional[Callback], optional
            callbacks used for training, 
            see `graphwar.training.callbacks`, by default None
        verbose : Optional[int], optional
            verbosity during training, can be:
            None, 1, 2, 3, 4, by default 1
        epochs : int, optional
            training epochs, by default 100

        """

        model = self.model
        model.stop_training = False
        validation = val_y is not None

        if validation:
            validation = True
            val_g = g if val_g is None else val_g
            val_data = self.config_test_data(val_g, val_y, val_index)

        # Setup callbacks
        self.callbacks = callbacks = self.config_callbacks(verbose, epochs, callbacks=callbacks)
        train_data = self.config_train_data(g, y, index)

        logs = BunchDict()

        if verbose:
            print("Training...")

        callbacks.on_train_begin()
        try:
            for epoch in range(epochs):
                callbacks.on_epoch_begin(epoch)
                train_logs = self.train_step(train_data)
                logs.update({k: self.to_item(v) for k, v in train_logs.items()})

                if validation:
                    valid_logs = self.test_step(val_data)
                    logs.update({("val_" + k): self.to_item(v) for k, v in valid_logs.items()})

                callbacks.on_train_batch_end(len(train_data), logs)
                callbacks.on_epoch_end(epoch, logs)

                if model.stop_training:
                    print(f"Early Stopping at Epoch {epoch}", file=sys.stderr)
                    break

        finally:
            callbacks.on_train_end()

        return self

    def train_step(self, dataloader: DataLoader) -> dict:
        """One-step training on the input dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            the trianing dataloader

        Returns
        -------
        dict
            the output logs, including `loss` and `val_accuracy`, etc.
        """
        optimizer = self.optimizer
        loss_fn = self.loss
        model = self.model

        optimizer.zero_grad()
        self.reset_metrics()
        model.train()

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)
            if not isinstance(x, tuple):
                x = x,
            out = model(*x)[out_index]
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))

    def evaluate(self, g: DGLGraph, y: Optional[Tensor] = None,
                 index: Optional[Tensor] = None, verbose: Optional[int] = 1) -> BunchDict:
        """Simple evaluation step for `:attr:model`

        Parameters
        ----------
        g : DGLGraph
            the dgl graph used for evaluation
        y : Optional[Tensor], optional
            the evaluation labels, by default None
        index : Optional[Tensor], optional
            the evaluation index/mask, such as testing nodes 
            index or mask, by default None
        verbose : Optional[int], optional
            verbosity during evaluation, by default 1

        Returns
        -------
        BunchDict
            the dict-like output logs
        """
        if verbose:
            print("Evaluating...")

        test_data = self.config_test_data(g, y=y, index=index)
        progbar = Progbar(target=len(test_data),
                          verbose=verbose)
        logs = BunchDict(**self.test_step(test_data))
        logs.update({k: self.to_item(v) for k, v in logs.items()})
        progbar.update(len(test_data), logs)
        return logs

    @torch.no_grad()
    def test_step(self, dataloader: DataLoader) -> dict:
        loss_fn = self.loss
        model = self.model
        model.eval()
        callbacks = self.callbacks
        self.reset_metrics()

        for epoch, batch in enumerate(dataloader):
            callbacks.on_test_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)
            if not isinstance(x, tuple):
                x = x,
            out = model(*x)[out_index]
            loss = loss_fn(out, y)
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            callbacks.on_test_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def predict_step(self, dataloader: DataLoader) -> Tensor:
        model = self.model
        model.eval()
        outs = []
        callbacks = self.callbacks
        for epoch, batch in enumerate(dataloader):
            callbacks.on_predict_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            if not isinstance(x, tuple):
                x = x,
            out = model(*x)[out_index]
            outs.append(out)
            callbacks.on_predict_batch_end(epoch)

        return torch.cat(outs, dim=0)

    def predict(self, g: DGLGraph, index: Optional[Tensor] = None,
                transform: Callable = torch.nn.Softmax(dim=-1)) -> Tensor:
        predict_data = self.config_test_data(g, y=None, index=index)
        out = self.predict_step(predict_data).squeeze()
        if transform is not None:
            out = transform(out)
        return out

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 5e-4)
        return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def config_loss(self) -> Callable:
        return torch.nn.CrossEntropyLoss()

    def config_metrics(self) -> Callable:
        return Accuracy()

    def config_callbacks(self, verbose, epochs, callbacks=None) -> Callback:
        callbacks = CallbackList(callbacks=callbacks, add_history=True, add_progbar=True if verbose else False)
        callbacks.set_model(self.model)
        callbacks.set_params(dict(verbose=verbose, epochs=epochs))
        return callbacks

    def config_train_data(self, g: DGLGraph, y: Optional[Tensor] = None, index: Optional[Tensor] = None) -> DataLoader:
        g, y, index = self.to_device((g, y, index))
        feat = g.ndata.get(_FEATURE, None)
        dataset = ((g, feat), y, index)
        return DataLoader([dataset], batch_size=None, collate_fn=lambda x: x)

    def config_test_data(self, g: DGLGraph, y: Optional[Tensor] = None, index: Optional[Tensor] = None) -> DataLoader:
        return self.config_train_data(g, y=y, index=index)

    @property
    def metrics_names(self) -> List[str]:
        assert self.metrics is not None
        return ['loss'] + [metric.name for metric in self.metrics]

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        assert m is None or isinstance(m, torch.nn.Module)
        self._model = m

    def reset_metrics(self):
        if self.metrics is None:
            return
        for metric in self.metrics:
            metric.reset_states()

    @staticmethod
    def unravel_batch(batch):
        inputs = labels = out_index = None
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
            labels = batch[1]
            if len(batch) > 2:
                out_index = batch[-1]
        else:
            inputs = batch

        if isinstance(labels, (list, tuple)) and len(labels) == 1:
            labels = labels[0]
        if isinstance(out_index, (list, tuple)) and len(out_index) == 1:
            out_index = out_index[0]

        return inputs, labels, out_index

    @staticmethod
    def to_item(value: Any) -> Any:
        """Transform value to Python object

        Parameters
        ----------
        value : Any
            Tensor or Numpy array

        Returns
        -------
        Any
            Python object

        Example
        -------
        >>> x = torch.tensor(1.)
        >>> to_item(x)
        1.
        """
        if value is None:
            return value

        elif hasattr(value, 'numpy'):
            value = value.numpy()

        if hasattr(value, 'item'):
            value = value.item()

        return value

    def to_device(self, x: Any) -> Any:
        """Put `x` into the device `self.device`.

        Parameters
        ----------
        x : any object, probably `torch.Tensor`.
            the input variable used for model.

        Returns
        -------
        x : any object, probably `torch.Tensor`.
            the input variable that in the device `self.device`.
        """
        device = self.device

        def wrapper(inputs):
            if isinstance(inputs, tuple):
                return tuple(wrapper(input) for input in inputs)
            elif isinstance(inputs, dict):
                for k, v in inputs.items():
                    inputs[k] = wrapper(v)
                return inputs
            else:
                return inputs.to(device) if hasattr(inputs, 'to') else inputs

        return wrapper(x)
    
    def cache_clear(self):
        if hasattr(self.model, 'cache_clear'):
            self.model.cache_clear()
        return self
    
