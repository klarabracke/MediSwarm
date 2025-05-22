from datetime import datetime
import logging
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.datamodules import DataModule
from data.datasets import MiniDatasetForTesting
from models_for_testing import MiniCNNForTesting

from collections import Counter


class LogitCalibratedLoss(nn.Module):
    def __init__(self, tau, clients_label_counts, client_id):
        super().__init__()
        self.tau = tau
        self.clients_label_counts = clients_label_counts
        self.client_id = client_id

    def forward(self, logits, targets):
        device = logits.device
        label_counts = self.clients_label_counts[self.client_id].to(device)
        targets = targets.to(device).long().view(-1)  # Wichtig: 1D LongTensor

        cal_logit = torch.exp(
            logits - (
                self.tau
                * torch.pow(label_counts, -1 / 4)
                .unsqueeze(0)
                .expand(logits.shape[0], -1)
            )
        )
        y_logit = torch.gather(cal_logit, dim=-1, index=targets.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
        return loss.mean()


def load_environment_variables():
    return {
        'scratch_dir': os.getenv('SCRATCH_DIR', '/scratch/'),
        'max_epochs': int(os.getenv('MAX_EPOCHS', 100)),
        'min_peers': int(os.getenv('MIN_PEERS', 2)),
        'max_peers': int(os.getenv('MAX_PEERS', 7)),
        'use_adaptive_sync': os.getenv('USE_ADAPTIVE_SYNC', 'False').lower() == 'true',
        'sync_frequency': int(os.getenv('SYNC_FREQUENCY', 1024)),
        'prediction_flag': os.getenv('PREDICT_FLAG', 'ext')
    }


def create_run_directory(scratch_dir):
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
    return os.path.join(scratch_dir, f"{current_time}_minimal_training_pytorch_cnn")


def set_up_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


def set_up_data_module(env_vars):
    ds = MiniDatasetForTesting()
    labels = ds.get_labels()
    indices = list(range(len(ds)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

    ds_train = Subset(ds, train_indices)
    ds_val = Subset(ds, val_indices)

    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=1,
        num_workers=16,
        pin_memory=True,
    )
    return dm, ds_train


def prepare_training(logger):
    env_vars = load_environment_variables()
    path_run_dir = create_run_directory(env_vars['scratch_dir'])
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")

    accelerator = 'gpu'
    logger.info(f"Using {accelerator} for training")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_module, ds_train = set_up_data_module(env_vars)

    train_targets = [ds_train[i]['target'] for i in range(len(ds_train))]
    num_classes = len(set(train_targets))
    counter = Counter(train_targets)
    label_counts_vec = [counter.get(i, 1e-8) for i in range(num_classes)]
    clients_label_counts = [torch.tensor(label_counts_vec, dtype=torch.float32).to(device)]

    client_id = 0
    tau = 1.0

    loss_fn = LogitCalibratedLoss(tau, clients_label_counts, client_id)
    model = MiniCNNForTesting(loss=loss_fn)
    model = model.to(device)

    to_monitor = "val/AUC_ROC"
    min_max = "max"
    log_every_n_steps = 1

    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor=to_monitor,
        save_last=True,
        save_top_k=2,
        mode=min_max,
    )

    trainer = Trainer(
        accelerator=accelerator,
        precision=16,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps,
        max_epochs=2,
        num_sanity_val_steps=2,
        logger=TensorBoardLogger(save_dir=path_run_dir)
    )

    return data_module, model, checkpointing, trainer


def validate_and_train(logger, data_module, model, trainer):
    logger.info("--- Validate global model ---")
    trainer.validate(model, datamodule=data_module)

    logger.info("--- Train new model ---")
    trainer.fit(model, datamodule=data_module)


def finalize_training(logger, model, checkpointing, trainer):
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
    logger.info('Training completed successfully')