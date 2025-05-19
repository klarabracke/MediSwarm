from typing import List, Union
from pathlib import Path
import json
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy

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

    def _epoch_end(self, outputs, state): return

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        self._step_train += 1
        return self._step(batch, batch_idx, "train", self._step_train, optimizer_idx)
    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val, optimizer_idx)
    def test_step(self, batch, batch_idx, optimizer_idx=0):
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test, optimizer_idx)
    def training_epoch_end(self, outputs): return
    def validation_epoch_end(self, outputs): return
    def test_epoch_end(self, outputs): return

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
        from pytorch_lightning.utilities.cloud_io import load as pl_load, pl_legacy_patch
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
            loss_kwargs={},
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-2},
            lr_scheduler=None,
            lr_scheduler_kwargs={},
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

        self.auc_roc = nn.ModuleDict({
            state: AUROC(**(aucroc_kwargs or {"task": "binary"})).cpu()
            for state in ["train_", "val_", "test_"]
        })
        self.acc = nn.ModuleDict({
            state: Accuracy(**(acc_kwargs or {"task": "binary"})).cpu()
            for state in ["train_", "val_", "test_"]
        })

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx: int):
        source, target = batch['source'].cpu(), batch['target'].cpu()
        target_for_loss = target.float().view(-1, 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        source_on_device = source.to(device)
        pred = self(source_on_device)
        if pred.dtype != torch.float32:
            pred = pred.to(torch.float32)
        pred = pred.cpu()
        batch_size = source.shape[0]
        logging_dict = {}
        try:
            loss_val = self.loss(pred.to(device), target_for_loss.to(device))
            logging_dict['loss'] = loss_val
        except Exception as e:
            print("[ERROR] Loss computation failed:", str(e))
            raise

        tm_pred = pred.squeeze(-1)
        tm_target = target.view(-1).long()
        tm_pred_prob = torch.sigmoid(tm_pred)

        if tm_target.numel() == 1 or len(torch.unique(tm_target)) < 2:
            print("[WARNING] Skipping metric computation: only one class in batch!")
        else:
            acc_value = Accuracy(task="binary", threshold=0.0).cpu()(tm_pred, tm_target)
            auroc_value = AUROC(task="binary").cpu()(tm_pred_prob, tm_target)
            self.log(f"{state}/ACC", acc_value, batch_size=batch_size, on_step=False, on_epoch=True)
            self.log(f"{state}/AUC_ROC", auroc_value, batch_size=batch_size, on_step=False, on_epoch=True)

        return loss_val