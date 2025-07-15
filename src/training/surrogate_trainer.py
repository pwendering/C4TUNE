"""

Surrogate model trainer

"""

from .base_trainer import BaseTrainer
import torch
from losses.losses import mse_loss, MSEMonotonicityLoss
from utils.input_transform import prepare_model_inputs

class SurrogateTrainer(BaseTrainer):
    
    
    def __init__(self, model, dataloaders, train_config, device):
        super().__init__(model, dataloaders, train_config, device)
        
    
    def train(self):
        '''
        
        Surrogate model training
            
        Returns
        -------
        loss : float
            final training loss of current epoch

        '''
        
        size = len(self.train_loader.dataset)
        
        self.model.train()
            
        co2_steps = self.train_loader.dataset.dataset.co2_steps
        light_a_co2 = self.train_loader.dataset.dataset.light_a_co2
        
        light_steps = self.train_loader.dataset.dataset.light_steps
        co2_a_light = self.train_loader.dataset.dataset.co2_a_light
        
        # averages and standard deviations of parameters in training set
        p_av = self.data_stats['train']['p_av']
        p_sd = self.data_stats['train']['p_sd']
        
        for batch, (y, X, _) in enumerate(self.train_loader):
            
            b_size = X.size()[0]
            
            self.optimizer.zero_grad()
            
            # transfer data to device
            X, y = X.to(self.device), y.to(self.device)
            
            # standardize parameters
            X_norm = (X - p_av) / p_sd
            
            _, co2_steps_rep, light_a_co2_rep, _, light_steps_rep, co2_a_light_rep = \
                prepare_model_inputs(
                    None, co2_steps, light_a_co2,
                    None, light_steps, co2_a_light,
                    b_size)
     
            # Forward pass to obtain curve predictions
            predictions = self.model(
                X_norm, [
                    torch.cat((co2_steps_rep, light_a_co2_rep), dim=-1),
                    torch.cat((light_steps_rep, co2_a_light_rep), dim=-1)
                    ]
                )
            
            # compute loss
            loss = self.loss_fn(torch.cat((predictions[0], predictions[1]), 1), y)
            
            # backpropagation
            loss.backward()
            
            # gradient clipping
            if self.config.training.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.training.gradient_clipping_max,
                    norm_type=self.config.training.gradient_clipping_norm)
            
            self.optimizer.step()
            
            if batch % self.config.training.display_step == 0:
                tmp_loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss ({self.config.training.loss_fn}): {tmp_loss:.4f} [{current:<6d}/{size:<5d}]")
               
        return loss
    
    def test(self):
        '''
        
        Surrogate model testing
        
        Returns
        -------
        test_losses : dict
            test errors of current epoch

        '''

        # number of batches
        num_batches = len(self.test_loader)
        
        # evaluation mode
        self.model.eval()
        
        # initialize different testing statistics
        test_loss = 0
        mse = 0
        combined_error = 0
        
        co2_steps = self.test_loader.dataset.dataset.co2_steps
        light_a_co2 = self.test_loader.dataset.dataset.light_a_co2
        
        light_steps = self.test_loader.dataset.dataset.light_steps
        co2_a_light = self.test_loader.dataset.dataset.co2_a_light
        
        # averages and standard deviations of parameters in training set
        p_av = self.data_stats['test']['p_av']
        p_sd = self.data_stats['test']['p_sd']
        
        with torch.no_grad():
            
            for y, X, _ in self.test_loader:
                
                b_size = X.size()[0]
                
                X, y = X.to(self.device), y.to(self.device)
                
                # standardize parameters
                X_norm = (X - p_av) / p_sd
                
                _, co2_steps_rep, light_a_co2_rep, _, light_steps_rep, co2_a_light_rep = \
                    prepare_model_inputs(
                        None, co2_steps, light_a_co2,
                        None, light_steps, co2_a_light,
                        b_size)
                
                pred = self.model(
                    X_norm,
                    [
                        torch.cat((co2_steps_rep, light_a_co2_rep), dim=-1),
                        torch.cat((light_steps_rep, co2_a_light_rep), dim=-1)
                    ]
                )
                
                test_loss += self.loss_fn(torch.cat((pred[0], pred[1]), 1), y).item()
                mse += mse_loss(torch.cat((pred[0], pred[1]), 1), y).item()
                combined_error += MSEMonotonicityLoss(len(co2_steps))(torch.cat((pred[0], pred[1]), 1), y).item()
                
        test_loss /= num_batches
        mse /= num_batches
        combined_error /= num_batches
        
        print(f"Test error: {self.config.training.loss_fn} = {test_loss:.4f}")
        print(f"Test error: MSE = {mse:.4f}")
        print(f"Test error: Combined error = {combined_error:.4f}")
        
        return {"test_loss": test_loss, "mse": mse, "combined_error": combined_error}