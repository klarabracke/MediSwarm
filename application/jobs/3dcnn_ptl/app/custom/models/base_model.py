from typing import Any, List, Union
import torch
import torch.nn as nn
from torchmetrics import AUROC, Accuracy
from torch.optim import AdamW
import pytorch_lightning as pl


class LCalLoss(nn.Module):
   
   
    def __init__(self, alpha=0.5, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
       

        Args:
            pred (Tensor): Die Vorhersagen des Modells.
            target (Tensor): Die tatsächlichen Zielwerte.

        Returns:
            Tensor: Die berechnete Loss.
        """
       
        loss = self.alpha * torch.mean((pred - target) ** 2) + self.beta * torch.mean(torch.abs(pred - target))
        return loss


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


class BasicModel(VeryBasicModel):
    """
    Ein einfaches Modell mit Optimierer- und Learning-Rate-Scheduler-Konfigurationen.
    """
    def __init__(
            self,
            optimizer=AdamW,
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
            loss: nn.Module = LCalLoss, 
            loss_kwargs={},
            optimizer=AdamW,
            optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-2},
            lr_scheduler=None,
            lr_scheduler_kwargs={},
            aucroc_kwargs={"task": "binary"},
            acc_kwargs={"task": "binary"}
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
        """Schritt-Funktion für Training, Validierung und Test.

        Args:
            batch (dict): Eingabebatch.
            batch_idx (int): Batch-Index.
            state (str): Zustand des Modells ('train', 'val', 'test').
            step (int): Aktueller Schritt.
            optimizer_idx (int): Index des Optimierers.

        Returns:
            Tensor: Der Verlustwert.
        """
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
                self.log(f"{state}/{metric_name}", metric_val.cpu() if hasattr(metric_val, 'cpu') else metric_val,
                         batch_size=batch_size, on_step=True, on_epoch=True)

           
            self.log(f"{state}/ACC", self.acc[state + "_"].compute().cpu(), batch_size=batch_size, on_step=False, on_epoch=True)
            self.log(f"{state}/AUC_ROC", self.auc_roc[state + "_"].compute().cpu(), batch_size=batch_size, on_step=False, on_epoch=True)

           
            self.acc[state + "_"].reset()
            self.auc_roc[state + "_"].reset()

        return logging_dict['loss']
