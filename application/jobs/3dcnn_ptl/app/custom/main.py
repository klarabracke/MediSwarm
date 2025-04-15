#!/usr/bin/env python3

import sys
import os
import logging

# Setze den Pfad zum übergeordneten Verzeichnis, damit Module korrekt importiert werden
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import nvflare.client.lightning as flare
import nvflare.client as flare_util
import torch
import threedcnn_ptl_new

# Definiere die Trainingsmodi
TRAINING_MODE = os.getenv("TRAINING_MODE", "local_training") 
TM_PREFLIGHT_CHECK = "preflight_check"
TM_LOCAL_TRAINING = "local_training"
TM_SWARM = "swarm"

# Überprüfe den Trainingsmodus
if TRAINING_MODE == TM_SWARM:
    flare_util.init()
    SITE_NAME = flare.get_site_name()
    NUM_EPOCHS = threedcnn_ptl_new.get_num_epochs_per_round(SITE_NAME)
elif TRAINING_MODE in [TM_PREFLIGHT_CHECK, TM_LOCAL_TRAINING]:
    SITE_NAME = os.getenv("SITE_NAME", "default_site")  
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 10))  
else:
    raise Exception(f"Illegal TRAINING_MODE {TRAINING_MODE}")

def main():
    """
    Main function for training and evaluating the model using NVFlare and PyTorch Lightning.
    """
    # Logger einrichten
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Überprüfe und bereite das Training vor
    try:
        data_module, model, checkpointing, trainer, path_run_dir, env_vars = threedcnn_ptl_new.prepare_training(logger, NUM_EPOCHS, SITE_NAME)

        if TRAINING_MODE == TM_SWARM:
            flare.patch(trainer)  # Trainer patchen, um Swarm-Lernen zu ermöglichen
            torch.autograd.set_detect_anomaly(True)

            logger.info(f"Site name: {SITE_NAME}")

            # Swarm Training Loop
            while flare.is_running():
                input_model = flare.receive()
                logger.info(f"Current round: {input_model.current_round}")

                threedcnn_ptl_new.validate_and_train(logger, data_module, model, trainer)

        elif TRAINING_MODE in [TM_PREFLIGHT_CHECK, TM_LOCAL_TRAINING]:
            threedcnn_ptl_new.validate_and_train(logger, data_module, model, trainer)

        if TRAINING_MODE in [TM_LOCAL_TRAINING, TM_SWARM]:
            threedcnn_ptl_new.finalize_training(logger, model, checkpointing, trainer, path_run_dir, env_vars)

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
