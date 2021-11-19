import torch
from graphwar.training import Trainer


class RobustGCNTrainer(Trainer):

    def train_step(self, dataloader):
        loss_fn = self.loss
        model = self.model

        self.reset_metrics()
        model.train()
        
        kl = self.cfg.get('kl', 5e-4)

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)
            
            if not isinstance(x, tuple):
                x = x,
            out = model(*x)[out_index]
            # ================= add KL loss here =============================
            mean, var = model.mean, model.var
            kl_loss = -0.5 * torch.sum(torch.mean(1 + torch.log(var + 1e-8) -
                                                  mean.pow(2) + var, dim=1))
            loss = loss_fn(out, y) + kl * kl_loss
            # ===============================================================
            loss.backward()
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))
