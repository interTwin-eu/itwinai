import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class DataFrameDataset(Dataset):
    """ Dataset for loading data frames stored in a pickle file. """

    def __init__(self, filename):
        """
        Args:
            filename (str): Path to the pickle file containing the DataFrame.
        """
        self.dataframe = pd.read_pickle(filename)
        try:
            self.dataframe.applymap(lambda x: torch.tensor(x))
        except ValueError as e:
            raise ValueError("DataFrame must contain only numeric data") from e

    def __len__(self):
        """ Return the total number of samples in the data. """
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """ Retrieve the ith sample from the dataset, returning it as a tensor. """
        row = self.dataframe.iloc[idx]
        row_tensor = torch.tensor(row, dtype=torch.float32)
        return row_tensor


def create_data_loader(pickle_file, batch_size):
    dataset = DataFrameDataset(pickle_file)
    print(f'Total rows: {len(dataset)}')
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: x[0])
    return data_loader

pickle_file = 'file-499.pkl'
batch_size = 1
loader = create_data_loader(pickle_file, batch_size)

count = 0
for batch in loader:
    count += 1
    print(batch.shape)
    print(batch)
    if count > 1: 
        break

print(f"Processed {count} rows")