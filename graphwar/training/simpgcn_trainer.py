from graphwar.training import Trainer


class SimPGCNTrainer(Trainer):

    def train_step(self, dataloader):
        loss_fn = self.loss
        model = self.model

        self.reset_metrics()
        model.train()

        lambda_ = self.cfg.get("lambda_", 5.0)

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)

            if not isinstance(x, tuple):
                x = x,
            out, embeddings = model(*x)
            if out_index is not None:
                out = out[out_index]
            # ================= add regression loss here =============================
            loss = loss_fn(out, y) + lambda_ * \
                model.regression_loss(embeddings)
            # ===============================================================
            loss.backward()
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))
