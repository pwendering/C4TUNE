"""

Basic model trainer class

"""

import torch
from torch import FloatTensor
import torch.optim.lr_scheduler as lr_scheduler
import os
import time
import tempfile
from ray import train as raytrain
from ray.train import Checkpoint
from losses.losses import get_loss_function
import training.saver as saver
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    
    def __init__(self, model, dataloaders, train_config, device, surrogate_model=None):
        
        print("Configuring model trainer...")
        
        self.device = device
        self.model = model.to(device)
        self.config = train_config
        self.train_loader = dataloaders['train']
        self.test_loader = dataloaders['test']
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.loss_fn = self._init_loss(surrogate_model)
        self.data_stats = self._init_data_stats()
        
    def _init_optimizer(self):
        return torch.optim.SGD(self.model.parameters(),
                               lr=self.config.training.learning_rate,
                               weight_decay=self.config.training.weight_decay,
                               momentum=self.config.training.momentum,
                               nesterov=self.config.training.nesterov)
    
    def _init_scheduler(self):
        
        # linear scheduler
        f_lr = self.config.training.linear_decay_factor
        n_steps_lin = self.config.training.linear_scheduler_step
        linear_scheduler = lambda epoch: 1 - (epoch % n_steps_lin) * (1 - f_lr) / n_steps_lin
        scheduler_1 = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_scheduler)
        
        # stepwise scheduler
        n_steps_mult = self.config.training.stepwise_scheduler_step
        steplr_gamma = self.config.training.stepwise_decay_factor
        mult_scheduler = lambda epoch: steplr_gamma**(epoch//n_steps_mult)
        scheduler_2 = lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=mult_scheduler)
        
        return lr_scheduler.ChainedScheduler([scheduler_1, scheduler_2]) 
        
    def _init_loss(self, surrogate_model):
        return get_loss_function(self.config.training.loss_fn, self.config, surrogate_model)
        
    def _init_data_stats(self):
        
        
        idx_train = self.train_loader.dataset.indices
        idx_test = self.test_loader.dataset.indices
        
        data_stats = {
            "train": {
                "p_av": FloatTensor(self.train_loader.dataset.dataset.params.iloc[idx_train, :].mean().values),
                "p_sd": FloatTensor(self.train_loader.dataset.dataset.params.iloc[idx_train, ].std().values),
                "a_co2_av": FloatTensor(self.train_loader.dataset.dataset.a_co2.iloc[idx_train, :].mean().values),
                "a_co2_sd": FloatTensor(self.train_loader.dataset.dataset.a_co2.iloc[idx_train, :].std().values),
                "a_light_av": FloatTensor(self.train_loader.dataset.dataset.a_light.iloc[idx_train, :].mean().values),
                "a_light_sd": FloatTensor(self.train_loader.dataset.dataset.a_light.iloc[idx_train, :].std().values)
            },
            "test": {
                "p_av": FloatTensor(self.test_loader.dataset.dataset.params.iloc[idx_test, :].mean().values),
                "p_sd": FloatTensor(self.test_loader.dataset.dataset.params.iloc[idx_test, ].std().values),
                "a_co2_av": FloatTensor(self.test_loader.dataset.dataset.a_co2.iloc[idx_test, :].mean().values),
                "a_co2_sd": FloatTensor(self.test_loader.dataset.dataset.a_co2.iloc[idx_test, :].std().values),
                "a_light_av": FloatTensor(self.test_loader.dataset.dataset.a_light.iloc[idx_test, :].mean().values),
                "a_light_sd": FloatTensor(self.test_loader.dataset.dataset.a_light.iloc[idx_test, :].std().values)
                }
        }
         
        return data_stats
            
    def training_loop(self):
        
        # load previous checkpoint if training is resumed from a previous state
        if self.config.training.start_epoch > 1:
            resume_epoch = self.config.training.start_epoch - 1
            saver.resume(self.model, self.optimizer, self.scheduler,
                         os.path.join(self.config.paths.checkpoint_dir,
                         f"{self.config.model.name}-epoch-{resume_epoch}.pth"))
        
        test_losses = []
        
        print("=========== Training Neural Network ===========")
        
        for i in range(self.config.training.start_epoch, self.config.training.epochs+1):
            print(f"Epoch {i}\n -------------------------------")
            
            tstart = time.time()
            
            lr_before = self.optimizer.param_groups[0]["lr"]
            
            train_loss = self.train()
            
            test_errors = self.test()
            test_losses.append(test_errors['test_loss'])
            
            self.scheduler.step()
            lr_after = self.optimizer.param_groups[0]["lr"]
            
            tend = time.time()
            print(f"lr: {lr_before:.8f} -> {lr_after:.8f}")
            print(f"Time: {tend-tstart:.2f} s\n")
            
            if self.config.tuning.tune:
                
                # different checkpointing when running ray tune
                
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    checkpoint = None
                    
                    if i % self.config.checkpoint_every == 0:
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(temp_checkpoint_dir,
                                         f"{self.config.model.name}-epoch-{i}.pth")
                            )
                        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                        
                    raytrain.report(test_errors, checkpoint=checkpoint)
            else:
                if i % self.config.training.checkpoint_every == 0:
                    saver.checkpoint(
                        i, self.model, self.optimizer, self.scheduler,
                        train_loss, test_losses, str(self.loss_fn), 
                        os.path.join(self.config.paths.checkpoint_dir,
                        f"{self.config.model.name}-epoch-{i}.pth"))
        print("Done")
    
    @abstractmethod
    def train():
        pass
    
    @abstractmethod
    def test():
        pass   
    
    