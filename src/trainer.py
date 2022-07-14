from typing import Union, Dict, List

import numpy as np
from moge.model.metrics import Metrics
from pytorch_lightning import LightningModule
from torch.utils.data.distributed import DistributedSampler


class NodeClfTrainer(LightningModule):
    def __init__(self, hparams, dataset, metrics: Union[List[str], Dict[str, List[str]]], *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(metrics, dict):
            self.train_metrics = {
                subtype: Metrics(prefix="" + subtype + "_", loss_type=hparams.loss_type,
                                 n_classes=dataset.n_classes, multilabel=dataset.multilabel, metrics=keywords) \
                for subtype, keywords in metrics.items()}
            self.valid_metrics = {
                subtype: Metrics(prefix="val_" + subtype + "_", loss_type=hparams.loss_type,
                                 n_classes=dataset.n_classes, multilabel=dataset.multilabel, metrics=keywords) \
                for subtype, keywords in metrics.items()}
            self.test_metrics = {
                subtype: Metrics(prefix="test_" + subtype + "_", loss_type=hparams.loss_type,
                                 n_classes=dataset.n_classes, multilabel=dataset.multilabel, metrics=keywords) \
                for subtype, keywords in metrics.items()}
        else:
            self.train_metrics = Metrics(prefix="", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                         multilabel=dataset.multilabel, metrics=metrics)
            self.valid_metrics = Metrics(prefix="val_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                         multilabel=dataset.multilabel, metrics=metrics)
            self.test_metrics = Metrics(prefix="test_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                        multilabel=dataset.multilabel, metrics=metrics)
        hparams.name = self.name()
        hparams.inductive = dataset.inductive
        self._set_hparams(hparams)

    def name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            return self.__class__.__name__.replace("_", "-")

    def training_epoch_end(self, outputs):
        if isinstance(self.train_metrics, Metrics):
            metrics_dict = self.train_metrics.compute_metrics()
            self.train_metrics.reset_metrics()
        elif isinstance(self.train_metrics, dict):
            metrics_dict = {k: v for subtype, metrics in self.train_metrics.items() \
                            for k, v in metrics.compute_metrics().items()}

            for subtype, metrics in self.train_metrics.items():
                metrics.reset_metrics()

        self.log_dict(metrics_dict, prog_bar=True)

        return None

    def validation_epoch_end(self, outputs):
        if isinstance(self.valid_metrics, Metrics):
            metrics_dict = self.valid_metrics.compute_metrics()

            self.valid_metrics.reset_metrics()

        elif isinstance(self.valid_metrics, dict):
            metrics_dict = {k: v for subtype, metrics in self.valid_metrics.items() \
                            for k, v in metrics.compute_metrics().items()}

            for subtype, metrics in self.valid_metrics.items():
                metrics.reset_metrics()
        else:
            print("got here")

        self.log_dict(metrics_dict, prog_bar=True)

        return None

    def test_epoch_end(self, outputs):
        if isinstance(self.test_metrics, Metrics):
            metrics_dict = self.test_metrics.compute_metrics()
            self.test_metrics.reset_metrics()

        elif isinstance(self.test_metrics, dict):
            metrics_dict = {k: v for subtype, metrics in self.test_metrics.items() \
                            for k, v in metrics.compute_metrics().items()}

            for subtype, metrics in self.test_metrics.items():
                metrics.reset_metrics()

        self.log_dict(metrics_dict, prog_bar=True)
        return None

    def train_dataloader(self, **kwargs):
        if hasattr(self.hparams, "num_gpus") and self.hparams.num_gpus > 1:
            train_sampler = DistributedSampler(self.dataset.training_idx, num_replicas=self.hparams.num_gpus,
                                               rank=self.local_rank)
        else:
            train_sampler = None

        dataset = self.dataset.train_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size, batch_sampler=train_sampler,
                                                **kwargs)
        return dataset

    def val_dataloader(self, **kwargs):
        if hasattr(self.hparams, "num_gpus") and self.hparams.num_gpus > 1:
            train_sampler = DistributedSampler(self.dataset.validation_idx, num_replicas=self.hparams.num_gpus,
                                               rank=self.local_rank)
        else:
            train_sampler = None

        dataset = self.dataset.valid_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size, batch_sampler=train_sampler,
                                                **kwargs)

        return dataset

    def test_dataloader(self, **kwargs):
        if hasattr(self.hparams, "num_gpus") and self.hparams.num_gpus > 1:
            train_sampler = DistributedSampler(self.dataset.testing_idx, num_replicas=self.hparams.num_gpus,
                                               rank=self.local_rank)
        else:
            train_sampler = None

        dataset = self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                               batch_size=self.hparams.batch_size, batch_sampler=train_sampler,
                                               **kwargs)
        return dataset

    def get_n_params(self):
        model_parameters = filter(lambda tup: tup[1].requires_grad and "embedding" not in tup[0],
                                  self.named_parameters())

        params = sum([np.prod(p.size()) for name, p in model_parameters])
        return params
