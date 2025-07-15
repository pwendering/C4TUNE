
from src.data.data import PhotResponseDataset
import numpy as np
from torch import FloatTensor

def compute_data_stats(config):
    
    # read dataset
    dataset = PhotResponseDataset(config.datasets)
    
    # read train and test indices
    idx_train = np.load(config.datasets.train_idx)
    idx_test = np.load(config.datasets.test_idx)
    
    data_stats = {
        "train": {
            "p_av": FloatTensor(dataset.params.iloc[idx_train, :].mean().values),
            "p_sd": FloatTensor(dataset.params.iloc[idx_train, ].std().values),
            "a_co2_av": FloatTensor(dataset.a_co2.iloc[idx_train, :].mean().values),
            "a_co2_sd": FloatTensor(dataset.a_co2.iloc[idx_train, :].std().values),
            "a_light_av": FloatTensor(dataset.a_light.iloc[idx_train, :].mean().values),
            "a_light_sd": FloatTensor(dataset.a_light.iloc[idx_train, :].std().values)
        },
        "test": {
            "p_av": FloatTensor(dataset.params.iloc[idx_test, :].mean().values),
            "p_sd": FloatTensor(dataset.params.iloc[idx_test, ].std().values),
            "a_co2_av": FloatTensor(dataset.a_co2.iloc[idx_test, :].mean().values),
            "a_co2_sd": FloatTensor(dataset.a_co2.iloc[idx_test, :].std().values),
            "a_light_av": FloatTensor(dataset.a_light.iloc[idx_test, :].mean().values),
            "a_light_sd": FloatTensor(dataset.a_light.iloc[idx_test, :].std().values)
            }
    }
    
    return data_stats
    