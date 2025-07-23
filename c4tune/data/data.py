"""

Dataset for curve and parameter prediction training

"""

from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from torch import FloatTensor


class PhotResponseDataset(Dataset):
    ''' 
    
    Dataset containing sampled parameters, simulated A/CO2 and A/light curves, 
    and associated CO2 and light steps.
    
    Args:
        data_config (OmegaConf)
        
    data_config fields:
        param_file : str
            Path to CSV file containing parameter samples
        a_co2_file : str
            Path to CSV file containing simulated A/CO2 curves
        light_a_co2_file : str
            Path to TXT file containing the constant light intensitiy for A/CO2 
            curves
        a_light_file : str
            Path to CSV file containing simulated A/light curves
        co2_a_light_file : str
            Path to TXT file containing the constant CO2 partial pressure for
            A/light curves
    
    '''
    
    def __init__(self, data_config, n_rows=None):
        
        # === ODE model parameters ===
        self.params = pd.read_csv(
            data_config.parameters,
            dtype="float32",
            nrows=n_rows)
        
        # === Anet/CO2 curves ===
        
        # Anet
        self.a_co2 = pd.read_csv(
            data_config.a_co2,
            dtype="float32",
            nrows=n_rows
        )
        
        # CO2 steps
        self.co2_steps = np.array(self.a_co2.columns, dtype='int')
        co2_mean = self.co2_steps.mean()
        co2_std = self.co2_steps.std()
        
        # constant light
        self.light_a_co2 = pd.read_csv(
            data_config.light_a_co2,
            header=None
        ).values[0][0]
        
        # === Anet/light curves ===

        # Anet
        self.a_light = pd.read_csv(
            data_config.a_light,
            dtype="float32",
            nrows=n_rows
        )
        
        # light steps
        self.light_steps =  np.array(self.a_light.columns, dtype='int')
        light_mean = self.light_steps.mean()
        light_std = self.light_steps.std()
        
        # constant CO2
        self.co2_a_light = pd.read_csv(
            data_config.co2_a_light,
            header=None
        ).values[0][0]
        
        # === z-scale CO2 and light values ===
        self.co2_steps = self.zscore_columns(self.co2_steps)
        self.light_a_co2 = [( self.light_a_co2-light_mean ) / light_std]
        
        self.light_steps = self.zscore_columns(self.light_steps)
        self.co2_a_light = [( self.co2_a_light - co2_mean ) / co2_std]
        
        # === convert environmental inputs to tensors ===
        
        self.co2_steps = FloatTensor(self.co2_steps)
        self.light_a_co2 = FloatTensor(self.light_a_co2)
        self.light_steps = FloatTensor(self.light_steps)
        self.co2_a_light = FloatTensor(self.co2_a_light)
        
    def __len__(self):
        return len(self.params.index)
    
    def __getitem__(self, idx):
        response = pd.concat([
            self.a_co2.iloc[idx, :],
            self.a_light.iloc[idx, :]
            ], axis=0
        )
        params = self.params.iloc[idx, :]
        return response.to_numpy(), params.to_numpy(), idx
    
    def __remove__(self, idx):
        
        self.a_co2.drop(index=idx, inplace=True, errors='ignore')
        self.a_light.drop(index=idx, inplace=True, errors='ignore')
        self.params.drop(index=idx, inplace=True, errors='ignore')
        
        # Reset indices to maintain consistency
        self.a_co2.reset_index(drop=True, inplace=True)
        self.a_light.reset_index(drop=True, inplace=True)
        self.params.reset_index(drop=True, inplace=True)
    
    @staticmethod
    def zscore_columns(x):
        return ( x - x.mean() ) / x.std()


def create_train_test_sets(config, n_rows=None):
    '''
    Split dataset into training and testing set and return respective 
    DataLoaders for training and testing.
    
    Parameters
    ----------
    config : OmegaConfig
        contains file paths to create the dataset
    n_rows : int
        number of rows to read from dataset; If None, all rows are read.

    Returns
    -------
    dataloaders : dict
        'train' : torch.utils.data.DataLoader
            DataLoader for training data
        'test' : torch.utils.data.DataLoader
            DataLoader for testing data

    '''
    
    print("Loading the dataset...")
    
    dataset = PhotResponseDataset(config.paths.datasets, n_rows=n_rows)
    
    training_data, test_data = random_split(
        dataset,[config.training.train_pct, 1-config.training.train_pct])
    train_dataloader = DataLoader(training_data,
                                  batch_size=config.training.batch_size,
                                  shuffle=config.training.shuffle)
    test_dataloader = DataLoader(test_data,
                                 batch_size=config.training.batch_size,
                                 shuffle=config.training.shuffle)
    
    # write train and test indices to text file
    np.save(config.paths.datasets.train_idx, train_dataloader.dataset.indices)
    np.save(config.paths.datasets.test_idx, test_dataloader.dataset.indices)
    
    return {"train": train_dataloader, "test": test_dataloader}