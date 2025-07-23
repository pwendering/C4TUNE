'''

Neural Network training

'''

import os
import torch
from data.data import create_train_test_sets
from models.model_surrogate import SurrogateModel
from utils.utils import load_and_freeze_model
from utils.experiment_logger import ExperimentLogger
from utils.env_setup import set_training_environment, get_config
from utils.paths import PROJECT_ROOT
import argparse


def main(model_config_file, base_config_file):
    
    config = get_config(base_config_file, model_config_file)
    
    set_training_environment(config)
    
    with ExperimentLogger(config) as logger:
    
        device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        
        if config.model.name == "surrogate":
            from training.surrogate_trainer import SurrogateTrainer
            model = SurrogateModel(config.model)
            trainer = SurrogateTrainer(model, create_train_test_sets(config, n_rows=None),
                                       config, device)
        elif config.model.name == "c4tune":
            from models.model_c4tune import ParameterPredictionModel
            from training.c4tune_trainer import C4tuneTrainer
            model = ParameterPredictionModel(config.model)
            
            surrogate_model = load_and_freeze_model(
                SurrogateModel, config.model.surrogate,
                config.paths.surrogate_checkpoint, device)
            
            trainer = C4tuneTrainer(model, create_train_test_sets(config, n_rows=None),
                                       config, device, surrogate_model)
        else:
            raise ValueError(f"Unknown model type: {config.model.name}")
            
        trainer.training_loop()
    
if __name__ == "__main__":
    
    base_config_file = os.path.join(PROJECT_ROOT, "config/base.yaml")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config/c4tune.yaml"),
                        help='Path to model-specific YAML configuration file')
    args = parser.parse_args()
    
    main(args.config, base_config_file)
    
    
