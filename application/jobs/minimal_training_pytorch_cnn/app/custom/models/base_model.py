from typing import List, Union
from pathlib import Path
import json
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.migration import pl_legacy_patch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torchmetrics.functional import auroc, accuracy

class VeryBasicModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self._step_train = -1
        self._step_val = -1
        self._step_test = -1

    def forward(self, x_in):
        raise NotImplementedError

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx: int):
        raise NotImplementedError

    def _epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]], state: str):
        return

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0):
        self._step_train += 1
        return self._step(batch, batch_idx, "train", self._step_train, optimizer_idx)

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0):
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val, optimizer_idx)

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0):
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test, optimizer_idx)

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self._epoch_end(outputs, "train")
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self._epoch_end(outputs, "val")
        return super().validation_epoch_end(outputs)

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self._epoch_end(outputs, "test")
        return super().test_epoch_end(outputs)

    @classmethod
    def save_best_checkpoint(cls, path_checkpoint_dir, best_model_path):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'w') as f:
            json.dump({'best_model_epoch': Path(best_model_path).name}, f)

    @classmethod
    def _get_best_checkpoint_path(cls, path_checkpoint_dir, version=0, **kwargs):
        path_version = 'lightning_logs/version_' + str(version)
        with open(Path(path_checkpoint_dir) / path_version / 'best_checkpoint.json', 'r') as f:
            path_rel_best_checkpoint = Path(json.load(f)['best_model_epoch'])
        return Path(path_checkpoint_dir) / path_rel_best_checkpoint

    @classmethod
    def load_best_checkpoint(cls, path_checkpoint_dir, version=0, **kwargs):
        path_best_checkpoint = cls._get_best_checkpoint_path(path_checkpoint_dir, version)
        return cls.load_from_checkpoint(path_best_checkpoint, **kwargs)

    def load_pretrained(self, checkpoint_path, map_location=None, **kwargs):
        if checkpoint_path.is_dir():
            checkpoint_path = self._get_best_checkpoint_path(checkpoint_path, **kwargs)
        with pl_legacy_patch():
            if map_location is not None:
                checkpoint = pl_load(checkpoint_path, map_location=map_location)
            else:
                checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)
        return self.load_weights(checkpoint["state_dict"], **kwargs)

    def load_weights(self, pretrained_weights, strict=True, **kwargs):
        filter_fn = kwargs.get('filter', lambda key: key in pretrained_weights)
        init_weights = self.state_dict()
        pretrained_weights = {key: value for key, value in pretrained_weights.items() if filter_fn(key)}
        init_weights.update(pretrained_weights)
        self.load_state_dict(init_weights, strict=strict)
        return self

class BasicModel(VeryBasicModel):
    def __init__(
            self,
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-2},
            lr_scheduler=None,
            lr_scheduler_kwargs={},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]

class BasicClassifier(BasicModel):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            spatial_dims: int,
            loss=torch.nn.BCEWithLogitsLoss,
            loss_kwargs=None,
            optimizer=torch.optim.AdamW,
            optimizer_kwargs=None,
            lr_scheduler=None,
            lr_scheduler_kwargs=None,
            aucroc_kwargs=None,
            acc_kwargs=None
    ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.spatial_dims = spatial_dims
        if isinstance(loss, type):
            self.loss = loss(**loss_kwargs)
        else:
            self.loss = loss
        self.loss_kwargs = loss_kwargs

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx: int):
        source, target = batch['source'], batch['target']
        target_for_loss = target.float().view(-1, 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        source_on_device = source.to(device)
        pred = self(source_on_device)
        if pred.dtype != torch.float32:
            pred = pred.to(torch.float32)
        pred = pred.cpu()
        target = target.cpu()
        batch_size = source.shape[0]

        logging_dict = {}
        try:
            loss_val = self.loss(pred.to(device), target_for_loss.to(device))
            logging_dict['loss'] = loss_val
        except Exception as e:
            print("[ERROR] Loss computation failed:", str(e))
            raise

        # Torchmetrics/Metrics auf CPU, lokal – no state!
        tm_pred = pred.squeeze(-1)
        tm_target = target.view(-1).long()
        tm_pred_prob = torch.sigmoid(tm_pred)

        if tm_target.numel() == 1 or len(torch.unique(tm_target)) < 2:
            print("[WARNING] Skipping metric computation: only one class in batch!")
            self.last_acc = None
            self.last_auroc = None
        else:
            acc_value = accuracy(tm_pred, tm_target, task="binary", threshold=0.0).cpu()
            auroc_value = auroc(tm_pred_prob, tm_target, task="binary").cpu()
            self.last_acc = acc_value
            self.last_auroc = auroc_value

        
        if self.last_acc is not None:
            self.log(f"{state}/ACC", self.last_acc, batch_size=batch_size, on_step=False, on_epoch=True)
        if self.last_auroc is not None:
            self.log(f"{state}/AUC_ROC", self.last_auroc, batch_size=batch_size, on_step=False, on_epoch=True)

        return logging_dict['loss']

    