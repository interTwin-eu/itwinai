import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

class DataFrameDataset(Dataset):
    """ Dataset for dynamically loading data frames stored in pickle files across multiple subdirectories. """

    def __init__(self, root_folder):
        """
        Initialize the dataset to load data frames from all pickle files within the subfolders of the root folder.
        
        Args:
            root_folder (str): Path to the root directory containing subdirectories with pickle files.
        """
        # Initialize an empty list to hold all file paths
        self.file_paths = []
        
        # Traverse the directory structure to find all pickle files
        for subdir, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith('.pkl'):
                    self.file_paths.append(os.path.join(subdir, file))
        print(self.file_paths)
        # Load all dataframes from the collected file paths
        self.dataframes = [pd.read_pickle(fp) for fp in self.file_paths]
        self.data = pd.concat(self.dataframes, ignore_index=True)
        print("type of combined data frames is:")
        print(type(self.data))
        print(self.data)
        self.data.map(lambda x: torch.tensor(x))

    def __len__(self):
        """ Return the total number of samples in all loaded data. """
        return len(self.data)
    
    def __getitem__(self, idx):
        """ Retrieve the ith sample from the combined dataset, returning it as a tensor. """
        row = self.data.iloc[idx]
        print(type(row))
        print(row)
        row_tensor = torch.tensor(row, dtype=torch.float32)
        return row_tensor

def create_data_loader(root_folder, batch_size):
    dataset = DataFrameDataset(root_folder)
    print(f'Total rows: {len(dataset)}')
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: x[0])
    return data_loader

# root_folder = '/p/scratch/intertwin/datasets/virgo'
root_folder = '../data2'
batch_size = 1
loader = create_data_loader(root_folder, batch_size)

# count = 0
# for batch in loader:
#     count += 1
#     print(batch.shape)
#     if count > 1:
#         break

# print(f"Processed {count} rows")
