from typing import List, Union, Any
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import load as pl_load
from pytorch_lightning.utilities.migration import pl_legacy_patch
from torchmetrics import AUROC, Accuracy

# FineGrained Calibration
class FineGrainedCalibratedLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', temperature=1.0, n_y=1, n_i=1):
        super(FineGrainedCalibratedLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.temperature = temperature
        self.n_y = n_y  # Anzahl der positiven Instanzen (n_y)
        self.n_i = n_i  # Anzahl der negativen Instanzen (n_i)

    def forward(self, logits, labels):
        # Kalibrierung
        logits = logits / self.temperature  

        # Berechne Margin 
        margin_y = torch.log(self.n_y / (self.n_y + self.n_i))  # Margin für positive Klasse
        margin_i = torch.log(self.n_i / (self.n_y + self.n_i))  # Margin für negative Klasse

        # Adjustiere Logits durch Margin
        adjusted_logits = torch.where(labels == 1, logits + margin_y, logits + margin_i)

        
        ce_loss = F.binary_cross_entropy_with_logits(adjusted_logits, labels, weight=self.weight, reduction='none')

      
        if self.reduction == 'mean':
            return ce_loss.mean()
        elif self.reduction == 'sum':
            return ce_loss.sum()
        else:
            return ce_loss


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

    def _epoch_end(self, outputs: Union[Any, List[Any]], state: str):
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

    def training_epoch_end(self, outputs: Union[Any, List[Any]]) -> None:
        self._epoch_end(outputs, "train")
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs: Union[Any, List[Any]]) -> None:
        self._epoch_end(outputs, "val")
        return super().validation_epoch_end(outputs)

    def test_epoch_end(self, outputs: Union[Any, List[Any]]) -> None:
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
    def __init__(self, optimizer=torch.optim.AdamW, optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-2}, lr_scheduler=None, lr_scheduler_kwargs={}):
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
        loss=FineGrainedCalibratedLoss,
        loss_kwargs={},
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-2},
        lr_scheduler=None,
        lr_scheduler_kwargs={},
        aucroc_kwargs={"task": "binary"},
        acc_kwargs={"task": "binary"},
    ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.spatial_dims = spatial_dims
        self.loss = loss(**loss_kwargs)
        self.loss_kwargs = loss_kwargs

        self.auc_roc = nn.ModuleDict({state: AUROC(**aucroc_kwargs) for state in ["train_", "val_", "test_"]})
        self.acc = nn.ModuleDict({state: Accuracy(**acc_kwargs) for state in ["train_", "val_", "test_"]})

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx: int):
        source, target = batch['source'], batch['target']
        target = target[:, None].float()
        batch_size = source.shape[0]

        pred = self(source)
        logging_dict = {}
        logging_dict['loss'] = self.loss(pred, target)

        with torch.no_grad():
            self.acc[state + "_"].update(pred, target)
            self.auc_roc[state + "_"].update(pred, target)

            for metric_name, metric_val in logging_dict.items():
                self.log(f"{state}/{metric_name}", metric_val.cpu() if hasattr(metric_val, 'cpu') else metric_val, batch_size=batch_size, on_step=True, on_epoch=True)

            self.log(f"{state}/ACC", self.acc[state + "_"].compute().cpu(), batch_size=batch_size, on_step=False, on_epoch=True)
            self.log(f"{state}/AUC_ROC", self.auc_roc[state + "_"].compute().cpu(), batch_size=batch_size, on_step=False, on_epoch=True)

            self.acc[state + "_"].reset()
            self.auc_roc[state + "_"].reset()

        return logging_dict['loss']
