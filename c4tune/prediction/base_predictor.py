
import torch
from abc import ABC, abstractclassmethod
from src.utils.data_stats import compute_data_stats
import json
from pathlib import Path


class BasePredictor(ABC):
    
    def __init__(self, model, checkpoint_path, device, config):
        
        self.model = model.to(device)
        self.device = device
        self.load_weights(checkpoint_path)
        self.model.eval()
        self.config = config
        # self.data_stats = self._get_param_stats()
        
    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    @abstractclassmethod
    def predict(self, inputs):
        pass
    
    @abstractclassmethod
    def _prepare_input(self, inputs):
        pass
    
    def _postprocess(self, outputs):
        return outputs.detach().cpu().numpy()
    
    def _get_param_stats(self):
        return compute_data_stats(self.config.paths)
    
    def _load_or_create_static_inputs(self, env_input):
        cache_path = Path(self.config.paths.cache_file)
        if cache_path.exists():
            print(f"\tReading cache from {cache_path}.")
            with open(cache_path) as f:
                parameters = self.python_to_tensor(json.load(f))
        else:
            print(f"\tWriting cache to {cache_path}.")
            data_stats = self._get_param_stats()
            
            # write input parameters
            with open(self.config.paths.cache_file, "w") as f:
                parameters = {
                    "env_input": env_input,
                    "data_stats": data_stats
                }
                json.dump(self.tensor_to_python(parameters), f, indent=2)
        return parameters
    
    @staticmethod
    def tensor_to_python(obj):
        """
        To save a dict of Tensors
        """
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: BasePredictor.tensor_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [BasePredictor.tensor_to_python(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(BasePredictor.tensor_to_python(v) for v in obj)
        else:
            return obj
        
    @staticmethod
    def python_to_tensor(obj):
        """
        To convert loaded values to tensor
        """
        if isinstance(obj, list):
            return torch.tensor(obj)
        elif isinstance(obj, dict):
            return {k: BasePredictor.python_to_tensor(v) for k, v in obj.items()}
        else:
            return obj
        