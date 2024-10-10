import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datetime import datetime
import os

class DataFrameDataset(torch.utils.data.Dataset):
    """ Dataset for dynamically loading data frames stored in pickle files across multiple subdirectories. """

    def __init__(self, root_folder):
        """ Initialize dataset with the path to the folder containing the pickle files. """
        self.file_paths = []
        for subdir, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith('.pkl'):
                    self.file_paths.append(os.path.join(subdir, file))
        # Load data frames and concatenate into one large DataFrame
        self.data = pd.concat((pd.read_pickle(fp) for fp in self.file_paths), ignore_index=True)
        self.data = self.data.applymap(lambda x: torch.tensor(x, dtype=torch.float) if isinstance(x, (int, float, list, np.ndarray)) else x)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return torch.stack([v.squeeze() for v in row])

def create_data_loader(root_folder, batch_size):
    dataset = DataFrameDataset(root_folder)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class TimeSeriesDatasetSplitter:
    def __init__(self, root_folder, train_proportion=0.8, validation_proportion=0.2, test_proportion=0.0, rnd_seed=None, name=None):
        self.train_proportion = train_proportion
        self.validation_proportion = validation_proportion
        self.test_proportion = test_proportion
        self.rnd_seed = rnd_seed
        self.name = name
        self.root_folder = root_folder
        self.loader = create_data_loader(self.root_folder, batch_size=1)  # Batch size can be adjusted as needed

    def get_or_load(self):
        """ Collects and concatenates data from all DataLoader batches. """
        data = []
        for batch in self.loader:
            data.append(batch)
        return pd.concat(data, ignore_index=True)

    def execute(self):
        """ Splits the loaded dataset into training and validation sets. """
        dataset = self.get_or_load()

        X_train, X_test, y_train, y_test = train_test_split(
            dataset.drop('target', axis=1), dataset['target'],
            test_size=self.validation_proportion, random_state=self.rnd_seed)

        return (X_train, y_train), (X_test, y_test), None

# Example usage
root_folder = '/path/to/your/root/folder'
splitter = TimeSeriesDatasetSplitter(root_folder=root_folder)
train_data, test_data, _ = splitter.execute()
print("Train Data:", train_data)
print("Test Data:", test_data)
