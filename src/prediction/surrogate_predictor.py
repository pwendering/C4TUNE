
import torch
from src.prediction.base_predictor import BasePredictor
from src.utils.input_transform import prepare_model_inputs
import pandas as pd
import numpy as np


class SurrogatePredictor(BasePredictor):
    
    def predict(self, parameters, env_input):
        
        self.static_inputs = self._load_or_create_static_inputs(env_input)
        
        with torch.no_grad():
            X = self._prepare_input(parameters)
            output = self.model(**X)
        
        return [self._postprocess(x) for x in output]
            
    def _prepare_input(self, parameters):
        
        env_input = self.static_inputs['env_input']
        data_stats = self.static_inputs['data_stats']
        
        if isinstance(parameters, pd.DataFrame) or isinstance(parameters, np.ndarray):
            parameters = torch.FloatTensor(parameters)
        
        for k in env_input.keys():
            if isinstance(env_input[k], pd.DataFrame) or isinstance(env_input[k], np.ndarray):
                env_input[k] = torch.FloatTensor(env_input[k])
        
        _, co2_steps, light_a_co2, _, light_steps, co2_a_light = \
            prepare_model_inputs(
                None, env_input['co2_steps'], env_input['light_a_co2'],
                None, env_input['light_steps'], env_input['co2_a_light'],
                parameters.shape[0])
        
        X = {
            "parameters": (parameters-data_stats['test']['p_av'])/data_stats['test']['p_sd'],
            "env_inputs": [
                torch.cat((co2_steps, light_a_co2), dim=-1),
                torch.cat((light_steps, co2_a_light), dim=-1)
                ]}
        
        return X
        
        
    
    