
import torch
from src.prediction.base_predictor import BasePredictor
from src.utils.input_transform import prepare_model_inputs
import pandas as pd
import numpy as np


class SurrogatePredictor(BasePredictor):
    
    def predict(self, parameters, env_input):
        
        if isinstance(parameters, pd.DataFrame) or isinstance(parameters, np.ndarray):
            parameters = torch.FloatTensor(parameters)
        
        for k in env_input.keys():
            if isinstance(env_input[k], pd.DataFrame) or isinstance(env_input[k], np.ndarray):
                env_input[k] = torch.FloatTensor(env_input[k])
        
        with torch.no_grad():
            X = self._prepare_input(parameters, env_input)
            output = self.model(**X)
        
        return [self._postprocess(x) for x in output]
            
    def _prepare_input(self, parameters, env_input):
        
        _, co2_steps, light_a_co2, _, light_steps, co2_a_light = \
            prepare_model_inputs(
                None, env_input['co2_steps'], env_input['light_a_co2'],
                None, env_input['light_steps'], env_input['co2_a_light'],
                parameters.shape[0])
        
        X = {
            "parameters": (parameters-self.data_stats['test']['p_av'])/self.data_stats['test']['p_sd'],
            "env_inputs": [
                torch.cat((co2_steps, light_a_co2), dim=-1),
                torch.cat((light_steps, co2_a_light), dim=-1)
                ]}
        
        return X
        
        
    
    