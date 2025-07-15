"""

Surrogate model

"""

import torch
from torch import nn

class Attention(nn.Module):
    '''
    
    Attention layer that calculates weights from parameter encoding and 
    environmental parameter encoding (CO2 or light steps and corresponding
                                      constant light and CO2 values)
    
    '''
    def __init__(self, param_encoding_dim, env_encoding_dim):
        '''
        Initialization for Attention

        Parameters
        ----------
        param_encoding_dim : int
            Dimension of generated parameter encoding
        env_encoding_dim : int
            Dimension of generated environmental input encoding

        '''
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(env_encoding_dim, param_encoding_dim)
        self.key_layer = nn.Linear(param_encoding_dim, param_encoding_dim)
        self.value_layer = nn.Linear(param_encoding_dim, param_encoding_dim)
    
    def forward(self, env_input, param_encoding):
        '''
        Forward pass through Attention layer

        Parameters
        ----------
        env_input : Tensor
            Environmental input.
        param_encoding : TYPE
            Generated parameter encoding.

        Returns
        -------
        weighted_values : Tensor
            Attention-weighted input values.

        '''
        
        # attention scores
        query = self.query_layer(env_input)
        key = self.key_layer(param_encoding).unsqueeze(1)
        attention_scores = torch.softmax(torch.bmm(query, key.transpose(1, 2)),
                                         dim = 1)
        # weighted values
        value = self.value_layer(param_encoding)
        weighted_values = attention_scores * value.unsqueeze(1)
        
        return weighted_values
    
class EnvInputBranch(nn.Module):
    '''
    
    Generate encoding of environmental input values using LSTM and Attention 
    with a final linear layer. The encoded parameters are injected at each step
    of the LSTM module.
    
    '''
    def __init__(self, n_input, env_encoding_dim, param_encoding_dim):
        '''
        Initialize EnvInputBranch

        Parameters
        ----------
        n_input : int
            Number of of environmental parameters.
        param_encoding_dim : int
            Dimension parameter encoding
        env_encoding_dim : int
            Dimension environmental input encoding

        '''
        super(EnvInputBranch, self).__init__()
        # LSTM module
        self.lstm = nn.LSTM(n_input+param_encoding_dim, env_encoding_dim,
                            batch_first=True)
        # Attention
        self.attention = Attention(param_encoding_dim, env_encoding_dim)
        # Final processing layer
        self.env_output = nn.Linear(env_encoding_dim + param_encoding_dim, 1)
        
    def forward(self, env_input, param_encoding):
        '''
        Forward pass through environmental-input-processing branch

        Parameters
        ----------
        env_input : Tensor
            concatenated environmental input (CO2 or light steps and constant
                                              light and CO2 values).
        param_encoding : Tensor
            Generated parameter encoding.

        Returns
        -------
        output : Tensor
            Processed combination of attention-weighted LSTM output and 
            parameter encoding, and the LSTM module output.

        '''
        # get batch size and number of light or CO2 steps
        batch_size, seq_len, _ = env_input.shape
        # repeat parameter encoding along inserted second dimension to 
        # adjust it as input for the LSTM module
        param_encoding_rep = param_encoding.unsqueeze(1).repeat(1, seq_len, 1)
        # Concatenated environmental input and parameter encoding for LSTM input
        lstm_input = torch.cat((env_input, param_encoding_rep), dim=-1)
        # Pass through LSTM layers
        lstm_out, _ = self.lstm(lstm_input)
        # Process LSTM output and parameter encoding with Attention layer
        attention_out = self.attention(lstm_out, param_encoding)
        # Concatenate LSTM output with attention-weighted parameter encoding
        combined_lstm_attention = torch.cat((lstm_out, attention_out), dim=-1)
        # Pass combined values through the final linear layer and remove second
        # dimension
        output = self.env_output(combined_lstm_attention).squeeze(-1)
        return output

class ParamInputBranch(nn.Module):
    '''
    
    Parameter-encoding branch
    
    '''
    def __init__(self, n_params, param_encoding_dim, n_hidden, activation_fun):
        '''
        Initialize ParamInputBranch

        Parameters
        ----------
        n_params : int
            Number of parameters in the ODE model.
        param_encoding_dim : int
            Dimension of hidden layers.
        n_hidden : int
            Number of hidden layers.
        activation_fun : str, optional
            Name of activation function. The default is "ReLU".

        '''
        super(ParamInputBranch, self).__init__()
        
        # initialize parameter encoding module
        self.param_encoding_layers = nn.Sequential()
        
        # First layer
        self.param_encoding_layers.add_module("param_encoding_layer_1", 
                                              nn.Linear(n_params, param_encoding_dim))
        # use specified activation function
        self.add_activation(activation_fun, 1)
        
        # Add more hidden layers followed by specified activation function
        for i in range(1, n_hidden):
            
            self.param_encoding_layers.add_module(
                f"param_encoding_layer_{i+1}",
                nn.Linear(param_encoding_dim, param_encoding_dim))
            self.add_activation(activation_fun, i+1)
        
    def forward(self, parameters):
        '''
        Forward pass through parameter encoding module

        Parameters
        ----------
        parameters : Tensor
            Raw ODE model parameters.

        Returns
        -------
        output : Tensor
            Generated parameter encoding.

        '''
        output = self.param_encoding_layers(parameters)
        return output
    
    def add_activation(self, activation_fun, i):
        '''
        Appends a specified activation function to param_encding_layers.

        Parameters
        ----------
        activation_fun : str
            Name of activation funtion.
        i : int
            Index of preceding layer; used for naming.

        '''
        match activation_fun:
            case "ReLU":
                self.param_encoding_layers.add_module(
                    f"{activation_fun}_{i}", nn.ReLU())
            case "LeakyReLU":
                self.param_encoding_layers.add_module(
                    f"{activation_fun}_{i}", nn.LeakyReLU())
            case "Sigmoid":
                self.param_encoding_layers.add_module(
                    f"{activation_fun}_{i}", nn.Sigmoid())
            case "LogSigmoid":
                self.param_encoding_layers.add_module(
                    f"{activation_fun}_{i}", nn.LogSigmoid())
            case _:
                self.param_encoding_layers.add_module(
                    f"{activation_fun}_{i}", nn.ReLU())

class SurrogateModel(nn.Module):
    '''
    
    Surrogate model, which directly predicts A/CO2 and A/light curves from ODE
    model parameters and environmental inputs, i.e., CO2 or light steps and 
    associated constant light and CO2 values.
    The parameters are passed through a series of dense layers with non-linear 
    activations to generate a parameter encoding. The environmental input 
    together with the parameter encoding are passed through an LSTM module.
    The LSTM output and parameter encoding are weighted using an Attention 
    layer. The LSTM output and attention-weighted parameter encoding are then
    concatenated and passed to a final linear layer, which reduces the
    dimension to the number of CO2 and light steps, respectively.
    Finally, the raw curve values are tranformed by a modified tanh activation 
    to match a range between -10 umol/m2/s and 70 umol/m2/s.
    
    Args:
        model_config (OmegaConf)
        
    model_config fields:
        param_encoding_dim : int
            Dimension of parameter encoding.
        n_hidden : int
            Number of hidden layers in parameter encoding module.
            Currently, all hidden layers will have the same dimension, which is
            given by param_encoding_dim.
        env_encoding_dim : int
            Dimension of environmental parameter encoding.
        n_co2_steps : int
            Number of CO2 steps used to generate A/CO2 curves.
        n_light_steps : int
            Number of light steps used to generate A/light curves.
        n_params : int
            Number of ODE model parameters.
        weight_init : str or None
            Name of weight initialization type or None of no specific weight 
            initialization should be applied.
        activation_fun : str
            Name of activation function used for all hidden layer in parameter
            encoding.
        
    
    '''
    def __init__(self, model_config):
        
        super(SurrogateModel, self).__init__()
        
        # set weight inialization type
        self.weight_init = model_config.weight_init
        
        # parameter encoding branch
        self.param_branch = ParamInputBranch(
            n_params=model_config.n_params,
            param_encoding_dim=model_config.param_encoding_dim,
            n_hidden=model_config.n_hidden,
            activation_fun=model_config.activation_fun
        )
        
        # CO2 and light input branches
        # The numbers of CO2 and light inputs are increased by 1, respectively,
        # to accomodate for constant light and CO2 values, which are the same 
        # across all steps.
        self.env_input_branches = nn.ModuleList([
            EnvInputBranch(model_config.n_co2_steps+1,
                           model_config.env_encoding_dim,
                           model_config.param_encoding_dim),
            EnvInputBranch(model_config.n_light_steps+1,
                           model_config.env_encoding_dim,
                           model_config.param_encoding_dim)
        ])
        
        # initialize weights
        self.apply(self.init_weights)
        
    def forward(self, parameters, env_inputs):
        '''
        Forward pass though surrogate model

        Parameters
        ----------
        parameters : Tensor
            Raw ODE model parameters.
        env_inputs : list
            list of Tensors, containing concatenated CO2 and light steps with 
            associated constant light and CO2 values.

        Returns
        -------
        outputs : list
            List of Tensors containing A/CO2 and A/light curve predictions.

        '''
        
        # generate parameter encoding
        param_encoding = self.param_branch(parameters)
        
        # generate curve prediction
        outputs = []
        for i, branch in enumerate(self.env_input_branches):
            # raw curve prediction
            pred_curve = branch(env_inputs[i], param_encoding)
            # apply range constraint
            pred_curve = 40 * torch.tanh(pred_curve) + 30
            
            outputs.append(pred_curve)
        
        return outputs
    
    def init_weights(self, m):
        '''
        Apply specified weight initialization to all linear layers in the 
        neural network.

        Parameters
        ----------
        m : SurrogateModel (nn.Module)
            Neural network.

        '''
        # weights
        if isinstance(m, nn.Linear):
            if self.weight_init == "Kaiming":
                torch.nn.init.kaiming_uniform_(m.weight)
            elif self.weight_init == "Xavier":
                torch.nn.init.xavier_uniform_(m.weight)
            elif self.weight_init == None:
                pass
            else:
                print("Unknown initialization function.")
                
            # biases
            m.bias.data.fill_(0.01)
            
    