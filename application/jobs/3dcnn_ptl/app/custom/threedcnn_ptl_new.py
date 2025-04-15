from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from collections import Counter
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data.datamodules import DataModule
from model_selector import select_model
from env_config import load_environment_variables, load_prediction_modules, prepare_dataset, generate_run_directory
import os
import logging


def get_num_epochs_per_round(site_name: str) -> int:
    NUM_EPOCHS_FOR_SITE = {
        "Krankenhaus 1": 3,
        "Krankenhaus 2": 5,
        "Krankenhaus 3": 7,
    }

   
    MAX_EPOCHS = NUM_EPOCHS_FOR_SITE.get(site_name, 5)

    print(f"Site name: {site_name}")
    print(f"Max epochs set to: {MAX_EPOCHS}")

    return MAX_EPOCHS


def set_up_logging():
   
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def set_up_data_module(ds_train, ds_val, logger):
   
    train_labels = [ds_train.dataset.get_labels()[i] for i in ds_train.indices]
    label_counts = Counter(train_labels)
    total_samples = len(train_labels)

    for label, count in label_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"Label '{label}': {percentage:.2f}% of the training set, Exact count: {count}")

    logger.info(f"Total number of different labels in the training set: {len(label_counts)}")

    # DataLoader für das Validierungsset 
    ads_val_data = DataLoader(ds_val, batch_size=2, shuffle=False)
    logger.info(f'ads_val_data type: {type(ads_val_data)}')
    logger.info(f'Train size: {len(ds_train)}')
    logger.info(f'Val size: {len(ds_val)}')

    
    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=1,
        num_workers=16,
        pin_memory=True,
    )
    return dm


def create_run_directory(env_vars):
    
    path_run_dir = generate_run_directory(
        env_vars['scratch_dir'],
        env_vars['task_data_name'],
        env_vars['model_name'],
        env_vars['local_compare_flag']
    )
    return path_run_dir


def run_cross_validation(site_name: str, logger):
    env_vars = load_environment_variables()
    ds, _ = prepare_dataset(env_vars['task_data_name'], env_vars['data_dir'], site_name=site_name)
    labels = ds.get_labels()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

   
    for fold, (train_idx, val_idx) in enumerate(skf.split(list(range(len(ds))), labels)):
        logger.info(f"Fold {fold+1}/5 for {site_name}")
        
        
        ds_train = Subset(ds, train_idx)
        ds_val = Subset(ds, val_idx)

       
        data_module = set_up_data_module(ds_train, ds_val, logger)
        
        max_epochs = get_num_epochs_per_round(site_name)
        
      
        model_name = env_vars['model_name']
        model = select_model(model_name)
        logger.info(f"Using model: {model_name}")

       
        path_run_dir = create_run_directory(env_vars)

       
        checkpointing = ModelCheckpoint(
            dirpath=str(path_run_dir),
            monitor="val/AUC_ROC",
            save_last=True,
            save_top_k=2,
            mode="max",
        )

    
        trainer = Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            precision=16,
            default_root_dir=str(path_run_dir),
            callbacks=[checkpointing],
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            max_epochs=max_epochs,
            num_sanity_val_steps=2,
            logger=TensorBoardLogger(save_dir=path_run_dir)
        )

     
        logger.info("Validate global model")
        trainer.validate(model, datamodule=data_module)

       
        logger.info("Train new model")
        trainer.fit(model, datamodule=data_module)

     
        model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)

        # Testvorhersagen laden und durchführen
        predict, prediction_flag = load_prediction_modules(env_vars['prediction_flag'])
        test_data_path = os.path.join(env_vars['data_dir'], env_vars['task_data_name'], 'test')
        if os.path.exists(test_data_path):
            predict(path_run_dir, test_data_path, env_vars['model_name'], last_flag=False, prediction_flag=prediction_flag)
        else:
            logger.info('No test data found, not running evaluation')

       
        logger.info(f"Fold {fold+1} for {site_name} completed successfully.")


if __name__ == "__main__":
    logger = set_up_logging()
    site_names = ["Krankenhaus 1", "Krankenhaus 2", "Krankenhaus 3"]

    # Cross-Validation für jeden Standort durchführen
    for site_name in site_names:
        run_cross_validation(site_name, logger)
        logger.info(f"Cross-validation for {site_name} completed successfully")
