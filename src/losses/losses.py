"""

Definitions of loss functions

"""

import torch
from torch import nn


def mean_relative_error_loss(pred, y):
    '''
    Calculates the mean average error between prediction and true values
    
    Parameters
    ----------
    pred : Tensor
        model prediction.
    y : Tensor
        true observations.

    Returns
    -------
    Tensor
        average relative error between predictions and observations.
    '''
    return torch.mean(torch.abs(pred-y)/y)

def mse_loss(pred, y):
    '''
    Calculate mean squared error loss

    Parameters
    ----------
    pred : Tensor
        predicted values
    y : Tensor
        true values

    Returns
    -------
    Tensor
        MSE.

    '''
    return torch.mean((pred-y)**2)

class MSEMonotonicityLoss(nn.Module):
    
    def __init__(self, n_co2):
        '''

        Parameters
        ----------
        n_co2 : int
            number of CO2 steps

        '''
        super(MSEMonotonicityLoss, self).__init__()
        self.n_co2 = n_co2
        
    def forward(self, pred, y):
        '''
        Calculates a combined error between predicted and true values, which is 
        based on the mean squared error (MSE). We allow one change in monotonicity
        within each A/CO2 or A/light curve. For each additional change, one MSE is
        added to the error term.
        

        Parameters
        ----------
        pred : Tensor
            predicted values
        y : Tensor
            true values
        
        Returns
        -------
        Tensor
            Combined error as explained above.

        '''
        
        # calculate MSE
        MSE = ((pred-y)**2).mean(dim=1)
        
        # variable that captures previous direction
        direction_old = torch.sign(torch.Tensor([1]).repeat(pred.shape[0]))
        
        # A/CO2
        # initialize counter for monotonicity changes
        c_co2 = torch.zeros(1, pred.shape[0])
        for i in range(1, self.n_co2):
            current_direction = pred[:, i] - pred[:, i-1]
            # increase counter of direction has changed from the previous value
            c_co2 += direction_old != torch.sign(current_direction)
            # update previous direction
            direction_old = torch.sign(current_direction)
        
        # A/light
        # initialize counter for monotonicity changes
        c_light = torch.zeros(1, pred.shape[0])
        direction_old = torch.sign(torch.Tensor([1]).repeat(pred.shape[0]))
        for i in range(self.n_co2+1, pred.shape[1]):
            current_direction = pred[:, i] - pred[:, i-1]
            # increase counter of direction has changed from the previous value
            c_light += direction_old != torch.sign(current_direction)
            # update previous direction
            direction_old = torch.sign(current_direction)
            
        # multiply MSE by average number of overstepped monotonicity change limit of 1
        MNT = MSE * (
            torch.cat((torch.zeros(1, c_co2.shape[1]), c_co2-1), dim=0).max(dim=0).values + 
            torch.cat((torch.zeros(1, c_light.shape[1]), c_light-1), dim=0).max(dim=0).values
        )
        
        return (MSE + MNT).mean()

class SimulationLoss(nn.Module):
    
    def __init__(self, surrogate_model):
        '''
    
        Parameters
        ----------
        surrogate_model : NeuralNetwork (nn.Module)
            Surrogate model that predicts A/CO2 and A/light curves based on
            a parameter set and CO2 and light inputs.
    
        '''
        super(SimulationLoss, self).__init__()
        self.surrogate_model = surrogate_model
                
    def forward(self, pred_params, true_params, true_curve_values, co2_input, co2_curve_light,
                   light_input, light_curve_co2, p_av, p_sd):
        '''
        Calculate the mean squared error between the true A/CO2 and A/light curves
        and the curves predicted by the surrogate model using the predicted parameters.

        Parameters
        ----------
        pred_params : Tensor
            predicted ODE model parameters.
        true_params : Tensor
            true ODE model parameters (ignored).
        true_curve_values : Tensor
            Concatenated A/CO2 and A/light curves.
        co2_input : Tensor
            CO2 steps used to measure A/CO2 curve.
        co2_curve_light : Tensor
            Constant light intensity value used measure A/CO2 curve.
        light_input : Tensor
            light intensity steps used to measure A/light curve.
        light_curve_co2 : Tensor
            Constant CO2 value used measure A/CO2 curve.
        p_av : Tensor
            Average parameter values.
        p_sd : Tensor
            Standard deviations of the parameters.

        Returns
        -------
        Tensor
            Mean squared error between true photosynthesis response curves and the
            ones predicted by the surrogate model using the predicted parameters.

        '''
        predicted_curves = self.surrogate_model((pred_params-p_av)/p_sd,
                           [torch.cat((co2_input, co2_curve_light), dim=-1),
                           torch.cat((light_input, light_curve_co2), dim=-1)])
        return torch.mean(torch.square(torch.sub(
                torch.cat((predicted_curves[0], predicted_curves[1]), dim=1),
                true_curve_values)))

class SimulationParameterCombLoss(nn.Module):
    
    def __init__(self, surrogate_model, reg_param):
        '''
        
        Parameters
        ----------
        surrogate_model : NeuralNetwork (nn.Module)
            Surrogate model that predicts A/CO2 and A/light curves based on
            a parameter set and CO2 and light inputs.
        reg_param : float, optional
            Regularization parameters to balance between the to error functions.
            The default is 1.0

        '''
        super(SimulationParameterCombLoss, self).__init__()
        self.simulation_loss = SimulationLoss(surrogate_model)
        self.reg_param = reg_param
        
    def forward(self, pred_params, true_params, true_curve_values, 
                p_av, p_sd, co2_input, co2_curve_light, light_input,
                light_curve_co2):
        '''
        Calculate the weighted sum of SimulationLoss and MeanRelativeErrorLoss with a
        regularization parameter.

        Parameters
        ----------
        pred_params : Tensor
            predicted ODE model parameters.
        true_params : Tensor
            true ODE model parameters.
        true_curve_values : Tensor
            Concatenated A/CO2 and A/light curves.
        p_av : Tensor
            Average parameter values.
        p_sd : Tensor
            Standard deviations of the parameters.
        co2_input : Tensor
            CO2 steps used to measure A/CO2 curve.
        co2_curve_light : Tensor
            Constant light intensity value used measure A/CO2 curve.
        light_input : Tensor
            light intensity steps used to measure A/light curve.
        light_curve_co2 : Tensor
            Constant CO2 value used measure A/CO2 curve.
        
            
        Returns
        -------
        Tensor
            Weighted sum of SimulationLoss and MeanRelativeErrorLoss based on the
            input parameters.

        '''
        mse_sim = self.simulation_loss(pred_params, None, true_curve_values, co2_input,
                                       co2_curve_light, light_input,
                                       light_curve_co2, p_av, p_sd)
        rel_error = mean_relative_error_loss(pred_params, true_params)
        
        return mse_sim + self.reg_param*rel_error

def get_loss_function(name, config=None, surrogate_model=None):
    '''
    Return the loss function specified by name.

    Parameters
    ----------
    name : str
        Name of the loss function.

    Returns
    -------
        loss function

    '''
    
    match name:
        case "MSELoss":
            return nn.MSELoss()
        case "L1Loss":
            return nn.L1Loss()
        case "simulation_loss":
            return SimulationLoss(surrogate_model)
        case "mse_monotonicity_loss":
            return MSEMonotonicityLoss(config.model.n_co2_steps)
        case "simulation_parameter_combined_loss":
            return SimulationParameterCombLoss(surrogate_model,
                                               config.training.reg_parameter)
    
    
    
    
    
    
    