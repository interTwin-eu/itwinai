import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class DataFrameDataset(Dataset):
    """ Dataset for dynamically loading data frames stored in pickle files. """

    def __init__(self, folder_path):
        """
        Initialize dataset with the path to the folder containing the pickle files.
        
        Args:
            folder_path (str): Path to the directory containing pickle files.
        """
        self.file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')]
        self.dataframes = []
        self.data_lengths = []
        
        for file_path in self.file_paths:
            df = pd.read_pickle(file_path)
            self.dataframes.append(df)
            self.data_lengths.append(len(df))

    def __len__(self):
        """ Return the total number of samples across all dataframes. """
        return sum(self.data_lengths)
    
    def __getitem__(self, idx):
        """ Retrieve the ith sample from the dataset, returning it as a tensor. """
        # Determine which file and index within the file the idx corresponds to
        for i, length in enumerate(self.data_lengths):
            if idx < length:
                self.dataframes[i].applymap(lambda x: torch.tensor(x))
                row = self.dataframes[i].iloc[idx]
                break
            idx -= length
        # Convert to tensor
        row_tensor = torch.tensor(row.to_numpy(), dtype=torch.float32)
        return row_tensor

def create_data_loader(folder_path, batch_size):
    dataset = DataFrameDataset(folder_path)
    print(f'Total samples: {len(dataset)}')
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: x[0])
    return data_loader

folder_path = './'
batch_size = 1
loader = create_data_loader(folder_path, batch_size)

count = 0
for batch in loader:
    count += 1
    print(batch.shape)
    print(batch)
    if count > 1: 
        break

print(f"Processed {count} rows")
