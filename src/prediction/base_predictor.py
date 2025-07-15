
import torch
from abc import ABC, abstractclassmethod
from src.utils.data_stats import compute_data_stats


class BasePredictor(ABC):
    
    def __init__(self, model, checkpoint_path, device, config):
        
        self.model = model.to(device)
        self.device = device
        self.load_weights(checkpoint_path)
        self.model.eval()
        self.config = config
        self.data_stats = self._get_param_stats()
        
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