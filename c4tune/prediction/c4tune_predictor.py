
import torch
from c4tune.prediction.base_predictor import BasePredictor
from c4tune.utils.input_transform import prepare_model_inputs
import pandas as pd
import numpy as np


class C4tunePredictor(BasePredictor):
    
    def predict(self, curves, env_input):
        
        self.static_inputs = self._load_or_create_static_inputs(env_input)
        
        with torch.no_grad():
            X = self._prepare_input(curves)
            output = self.model(**X)
        
        return self._postprocess(output, self.static_inputs['data_stats']['test']['p_av'])
            
    def _prepare_input(self, curves):
        
        env_input = self.static_inputs['env_input']
        data_stats = self.static_inputs['data_stats']
        for k in curves.keys():
            if isinstance(curves[k], pd.DataFrame) or isinstance(curves[k], np.ndarray):
                curves[k] = torch.FloatTensor(curves[k])
        
        for k in env_input.keys():
            if isinstance(env_input[k], pd.DataFrame) or isinstance(env_input[k], np.ndarray):
                env_input[k] = torch.FloatTensor(env_input[k])

        a_co2, co2_steps, light_a_co2, a_light, light_steps, co2_a_light = \
            prepare_model_inputs(
                (curves['a_co2']-data_stats['test']['a_co2_av'])/data_stats['test']['a_co2_sd'],
                env_input['co2_steps'], env_input['light_a_co2'],
                (curves['a_light']-data_stats['test']['a_light_av'])/data_stats['test']['a_light_sd'],
                env_input['light_steps'], env_input['co2_a_light'],
                curves['a_co2'].shape[0])
        
        X = {
            "a_co2_curve": torch.cat((a_co2, co2_steps, light_a_co2), dim=-1),
            "a_light_curve": torch.cat((a_light, light_steps, co2_a_light), dim=-1)
        }
        
        return X
    
    def _postprocess(self, output, p_av):
        x = super()._postprocess(output*p_av)
        return x
        