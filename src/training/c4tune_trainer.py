"""

C4TUNE model trainer

"""

from .base_trainer import BaseTrainer
import numpy as np
import torch
from torch import FloatTensor
from losses.losses import mean_relative_error_loss, SimulationLoss
from utils.input_transform import prepare_model_inputs

class C4tuneTrainer(BaseTrainer):
    
    def __init__(self, model, dataloaders, train_config, device, surrogate_model):
        super().__init__(model, dataloaders, train_config, device, surrogate_model)
        
        self.surrogate_model = surrogate_model
        self.simulation_loss = SimulationLoss(surrogate_model)
    
        self.L_train = FloatTensor(self.get_cholesky_L(
            np.log(self.train_loader.dataset.dataset.params)
        ))
        self.L_test = FloatTensor(self.get_cholesky_L(
            np.log(self.test_loader.dataset.dataset.params)
        ))
        
        # Write Cholesky decomposition of the logarithmic parameters to CSV files
        np.savetxt(train_config.paths.cholesky_train, self.L_train, delimiter=',')
        np.savetxt(train_config.paths.cholesky_test, self.L_test, delimiter=',')
    
    @staticmethod
    def get_cholesky_L(params):
        '''
        Get Cholesky decomposition of the covariance matrix of the parameter input    

        Parameters
        ----------
        params : pandas.core.frame.DataFrame
            ODE model parameters. The columns are expected to contain the variables
            with observations in the rows.

        Returns
        -------
        numpy.ndarray
            Cholesky decomposition of the parameter covariance matrix.

        '''
        
        return np.linalg.cholesky(np.cov(params, rowvar=False))
    
    
    def train(self):
        '''
        
        C4TUNE model training

        Returns
        -------
        loss : Tensor
            Current training loss.

        '''
        
        # size of dataset
        size = len(self.train_loader.dataset)
        
        # training mode
        self.model.train()
        
        # Set Cholesky decomposition of the covariance matrix
        self.model.L = self.L_train
        
        n_co2_steps = self.config.model.n_co2_steps
        co2_steps = self.train_loader.dataset.dataset.co2_steps
        light_a_co2 = self.train_loader.dataset.dataset.light_a_co2
        
        light_steps = self.train_loader.dataset.dataset.light_steps
        co2_a_light = self.train_loader.dataset.dataset.co2_a_light
        
        for batch, (X, y, _) in enumerate(self.train_loader):
            
            b_size = X.size()[0]
            
            self.optimizer.zero_grad()
            
            # transfer data to device
            X, y = X.to(self.device), y.to(self.device)
            
            a_co2, co2_steps_rep, light_a_co2_rep, a_light, light_steps_rep, co2_a_light_rep = \
                prepare_model_inputs(
                    (X[:, :n_co2_steps] - self.data_stats['train']['a_co2_av'])
                    / self.data_stats['train']['a_co2_sd'],
                    co2_steps, light_a_co2,
                    (X[:, n_co2_steps:] - self.data_stats['train']['a_light_av']) 
                    / self.data_stats['train']['a_light_sd'],
                    light_steps, co2_a_light, b_size)

            # forward pass to obtain prediction
            pred = self.model(
                torch.cat((a_co2, co2_steps_rep, light_a_co2_rep), dim=-1),
                torch.cat((a_light, light_steps_rep, co2_a_light_rep), dim=-1))
            
            # multiply predicted deviations with average parameter values from the 
            # training set
            pred = pred * self.data_stats['train']['p_av']
            
            # compute loss
            loss_args = {
                "true_curve_values": X,
                "co2_input": co2_steps_rep,
                "co2_curve_light": light_a_co2_rep,
                "light_input": light_steps_rep,
                "light_curve_co2": co2_a_light_rep,
                "p_av": self.data_stats['train']['p_av'],
                "p_sd": self.data_stats['train']['p_sd']
            }
            loss = self.loss_fn(pred, y, **loss_args)
            
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
        
        C4TUNE model testing
        
        Returns
        -------
        test_losses : dict
            test errors of current epoch
        

        '''
        # initialize different testing statistics
        test_loss = 0
        rel_error = 0
        rmse = 0
        sim_mse = 0
        
        # number of batches
        num_batches = len(self.test_loader)
        
        # evaluation mode
        self.model.eval()
        
        # Set Cholesky decomposition of the covariance matrix
        self.model.L = self.L_test
        
        n_co2_steps = self.config.model.n_co2_steps
        co2_steps = self.test_loader.dataset.dataset.co2_steps
        light_a_co2 = self.test_loader.dataset.dataset.light_a_co2
        
        light_steps = self.test_loader.dataset.dataset.light_steps
        co2_a_light = self.test_loader.dataset.dataset.co2_a_light
        
        with torch.no_grad():
            
            for X, y, _ in self.test_loader:
                
                b_size = X.size()[0]
                
                X, y = X.to(self.device), y.to(self.device)
                
                a_co2, co2_steps_rep, light_a_co2_rep, a_light, light_steps_rep, co2_a_light_rep = \
                    prepare_model_inputs(
                        (X[:, :n_co2_steps] - self.data_stats['test']['a_co2_av']) 
                        / self.data_stats['test']['a_co2_sd'],
                        co2_steps, light_a_co2,
                        (X[:, n_co2_steps:] - self.data_stats['test']['a_light_av'])
                        / self.data_stats['test']['a_light_sd'],
                        light_steps, co2_a_light, b_size)
        
                # forward pass to obtain prediction
                pred = self.model(
                    torch.cat((a_co2, co2_steps_rep, light_a_co2_rep), dim=-1),
                    torch.cat((a_light, light_steps_rep, co2_a_light_rep), dim=-1))
                
                # multiply predicted deviations with average parameter values
                pred = pred * self.data_stats['test']['p_av']
                
                # compute loss
                loss_args = {
                    "true_curve_values": X,
                    "co2_input": co2_steps_rep,
                    "co2_curve_light": light_a_co2_rep,
                    "light_input": light_steps_rep,
                    "light_curve_co2": co2_a_light_rep,
                    "p_av": self.data_stats['test']['p_av'],
                    "p_sd": self.data_stats['test']['p_sd']
                }
                test_loss += self.loss_fn(pred, y, **loss_args).item()
                        
                # calculate test error statistics
                rel_error += mean_relative_error_loss(pred, y).item()
                
                rmse += torch.sqrt(torch.mean((pred - y)**2)).item()
                
                sim_mse += self.simulation_loss(pred, y, **loss_args).item()
                
                
        test_loss /= num_batches
        rel_error /= num_batches
        sim_mse /= num_batches
        rmse /= num_batches
        
        print(f"Test error: {self.config.training.loss_fn} = {test_loss:.4f}")
        print(f"Test error: relative error (parameters) = {rel_error:<.4f}")
        print(f"Test error: RMSE (parameters) = {rmse:<.4f}")
        print(f"Test error: MSE (simulation) = {sim_mse:<.4f}")
        
        test_losses = {
            "test_loss": test_loss,
            "rel_error": rel_error,
            "sim_mse": sim_mse,
            "rmse": rmse
        }
        
        return test_losses