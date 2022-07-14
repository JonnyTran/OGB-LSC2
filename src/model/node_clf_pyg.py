import torch

from src.trainer import NodeClfTrainer


class GCN(NodeClfTrainer):
    def __init__(self, hparams, dataset, metrics=["accuracy"]) -> None:
        super().__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.node_types = dataset.node_types
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())
        self._name = f"GCN-{hparams.n_layers}-{hparams.t_order}"

        self.hparams.n_params = self.get_n_params()
        self.lr = self.hparams.lr

    def forward(self, **kwargs):
        pass

    def training_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X)

        loss = self.criterion.forward(y_pred, y_true, weights=weights)
        self.train_metrics.update_metrics(y_pred, y_true, weights=weights)

        self.log("loss", loss, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X)

        val_loss = self.criterion.forward(y_pred, y_true, weights=weights)
        self.valid_metrics.update_metrics(y_pred, y_true, weights=weights)

        self.log("val_loss", val_loss, )

        return val_loss

    def test_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=False)

        test_loss = self.criterion(y_pred, y_true, weights=weights)

        self.test_metrics.update_metrics(y_pred, y_true, weights=weights)
        self.log("test_loss", test_loss)

        return test_loss

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'alpha_activation', 'batchnorm', 'layernorm', "activation", "embeddings",
                    'LayerNorm.bias', 'LayerNorm.weight',
                    'BatchNorm.bias', 'BatchNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for name, p in param_optimizer \
                        if not any(key in name for key in no_decay) \
                        and "embeddings" not in name],
             'weight_decay': self.hparams.weight_decay if isinstance(self.hparams.weight_decay, float) else 0.0},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)],
             'weight_decay': 0.0},
        ]

        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.lr)

        extra = {}
        if "lr_annealing" in self.hparams and self.hparams.lr_annealing == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.num_training_steps,
                                                                   eta_min=self.lr / 100
                                                                   )
            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            print("Using CosineAnnealingLR", scheduler.state_dict())

        elif "lr_annealing" in self.hparams and self.hparams.lr_annealing == "restart":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                             T_0=50, T_mult=1,
                                                                             eta_min=self.lr / 100)
            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            print("Using CosineAnnealingWarmRestarts", scheduler.state_dict())

        return {"optimizer": optimizer, **extra}
