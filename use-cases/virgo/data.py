import os
from typing import Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from src.dataset import (
    generate_cut_image_dataset,
    generate_dataset_aux_channels,
    generate_dataset_main_channel,
    normalize_,
)
from torch.utils.data import DataLoader, Dataset
import numpy as np

class TimeSeriesDatasetGenerator(DataGetter):
    def __init__(
        self,
        data_root: str = "data",
        name: Optional[str] = None
    ) -> None:
        """Initialize the TimeSeriesDatasetGenerator class.

        Args:
            data_root (str): Root folder where datasets will be saved.
            name (Optional[str]): Name of the data getter component.
        """
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))
        self.data_root = data_root
        if not os.path.exists(data_root):
            os.makedirs(data_root, exist_ok=True)

    @monitor_exec
    def execute(self) -> pd.DataFrame:
        """Generate a time-series dataset, convert it to Q-plots,
        save it to disk, and return it.

        Returns:
            pd.DataFrame: dataset of Q-plot images.
        """
        df_aux_ts = generate_dataset_aux_channels(
            1000, 3, duration=16, sample_rate=500,
            num_waves_range=(20, 25), noise_amplitude=0.6
        )
        df_main_ts = generate_dataset_main_channel(
            df_aux_ts, weights=None, noise_amplitude=0.1
        )

        # save datasets
        save_name_main = 'TimeSeries_dataset_synthetic_main.pkl'
        save_name_aux = 'TimeSeries_dataset_synthetic_aux.pkl'
        df_main_ts.to_pickle(os.path.join(self.data_root, save_name_main))
        df_aux_ts.to_pickle(os.path.join(self.data_root, save_name_aux))

        # Transform to images and save to disk
        df_ts = pd.concat([df_main_ts, df_aux_ts], axis=1)
        df = generate_cut_image_dataset(
            df_ts, list(df_ts.columns),
            num_processes=20, square_size=64
        )
        save_name = 'Image_dataset_synthetic_64x64.pkl'
        df.to_pickle(os.path.join(self.data_root, save_name))
        return df

class DataFrameDataset(Dataset):
    def __init__(self, root_folder):
        self.file_paths = []
        self.file_paths = [os.path.join(dirpath, f)
                           for dirpath, dirs, files in os.walk(root_folder)
                           for f in files if f.endswith('.pkl')]
        print(self.file_paths)
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        dataframe = pd.read_pickle(self.file_paths[idx])
        return dataframe
    
def custom_collate(batch):
    # Combine all dataframes in the batch into a single dataframe
    return pd.concat(batch, ignore_index=True)


class SyntheticTimeSeriesDataset(Dataset):
    def __init__(self, root_folder: Optional[str] = None,):
        """Initialize the DataFrameDataset class.

        Args:
            root_folder (str): Location of the pickled DataFrames.
        """

        # Ensure the root folder exists
        if not os.path.isdir(root_folder):
            raise FileNotFoundError(f"Root folder '{root_folder}' not found.")

        # Find all file paths in root folder
        self.file_paths = [os.path.join(dirpath, f)
                           for dirpath, dirs, files in os.walk(root_folder)
                           for f in files if f.endswith('.pkl')]

    def __len__(self):
        """Return the total number of files in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """ Retrieve a data sample by index, convert to tensor, and normalize.

        Args:
            idx (int): Index of the file to retrieve.

        Returns:
            torch.Tensor: Concatenated and normalized data tensor
            of main and auxiliary channels.
        """
        # Load a single dataframe from the file
        dataframe = pd.read_pickle(self.file_paths[idx])

        # Convert all data in the DataFrame to torch tensors
        df = dataframe.map(lambda x: torch.tensor(x))

        # Divide Image dataset in main and aux channels.
        main_channel = list(df.columns)[0]
        aux_channels = list(df.columns)[1:]

        # Ensure that there are at least 3 auxiliary channels
        if len(aux_channels) < 3:
            print(f"Item with the index {idx} only has {len(aux_channels)} channels!")
            return None

        # Extract the main channel and auxiliary channels
        df_aux_all_2d = pd.DataFrame(df[aux_channels])
        df_main_all_2d = pd.DataFrame(df[main_channel])

        # Stack the main and auxiliary channels into 2D tensors
        signal_data_train_2d = torch.stack(
            [torch.stack(
                [df_main_all_2d[main_channel].iloc[i]]) for i in range(df_main_all_2d.shape[0])])

        aux_data_train_2d = torch.stack(
            [torch.stack(
                [df_aux_all_2d.iloc[i, 0],
                    df_aux_all_2d.iloc[i, 1],
                    df_aux_all_2d.iloc[i, 2]]
            ) for i in range(df_aux_all_2d.shape[0])]
        )

        # Concatenate the main and auxiliary channel tensors
        data_tensor = torch.cat([signal_data_train_2d, aux_data_train_2d], dim=1)

        # Normalize and return the concatenated tensor
        return normalize_(data_tensor)


class TimeSeriesDatasetSplitter(DataSplitter):
    def __init__(
        self,
        train_proportion: int | float,
        validation_proportion: int | float = 0.0,
        test_proportion: int | float = 0.0,
        rnd_seed: int | None = None,
        # images_dataset: str = "TimeSeries_dataset_synthetic_main.pkl",
        root_folder: str | None = None,
        name: str | None = None
    ) -> None:
        """Initialize the splitter for time-series datasets.

        Args:
            train_proportion (int | float): Proportion of files for the training set.
            validation_proportion (int | float): Proportion for validation.
            test_proportion (int | float): Proportion for testing.
            rnd_seed (int | None): Seed for randomization (optional).
            root_folder (str | None): Folder containing the dataset files.
            name (str | None): Name of the data splitter.
        """
        super().__init__(
            train_proportion,
            validation_proportion,
            test_proportion,
            name
        )
        self.save_parameters(**self.locals2params(locals()))
        self.rnd_seed = rnd_seed
        self.root_folder = root_folder

    @monitor_exec
    def execute(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Execute the dataset splitting process.

        Finds all pickled files in the root folder, then splits them into
        training, validation, and test sets based on the specified proportions.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: Training, validation, and test datasets.
        """

        whole_dataset = SyntheticTimeSeriesDataset(root_folder=self.root_folder)

        # Split file paths into train, validation, and test sets
        generator = torch.Generator().manual_seed(self.rnd_seed)
        [train_dataset, validation_dataset, test_dataset] = random_split(
            whole_dataset,
            [self.train_proportion,
             self.validation_proportion,
             self.test_proportion],
            generator=generator
        )
        print(f"Shape of item: {train_dataset.__getitem__(idx=5).shape}")
        return train_dataset, validation_dataset, test_dataset


class TimeSeriesDatasetSplitterSmall(DataSplitter):
    def __init__(
        self,
        train_proportion: int | float,
        validation_proportion: int | float = 0.0,
        test_proportion: int | float = 0.0,
        rnd_seed: Optional[int] = None,
        images_dataset: str = "data/Image_dataset_synthetic_64x64.pkl",
        name: Optional[str] = None
    ) -> None:
        """Class for splitting of smaller datasets. Use this class in the pipeline if the 
        entire dataset can fit into memory.

        Args:
            train_proportion (int | float): _description_
            validation_proportion (int | float, optional): _description_. Defaults to 0.0.
            test_proportion (int | float, optional): _description_. Defaults to 0.0.
            rnd_seed (Optional[int], optional): _description_. Defaults to None.
            images_dataset (str, optional): _description_. 
                Defaults to "data/Image_dataset_synthetic_64x64.pkl".
            name (Optional[str], optional): _description_. Defaults to None.
        """
        super().__init__(
            train_proportion, validation_proportion,
            test_proportion, name
        )
        self.save_parameters(**self.locals2params(locals()))
        self.validation_proportion = 1-train_proportion
        self.rnd_seed = rnd_seed
        self.name = name
        # self.images_dataset = images_dataset
        self.root_folder = root_folder
        self.loader = DataLoader(
            DataFrameDataset(root_folder),
            batch_size=300,
            collate_fn=custom_collate
                )

    def get_or_load(self):
        for combined_df in self.loader:
            return combined_df

    @monitor_exec
    def execute(
        self,
        dataset: Optional[pd.DataFrame] = None
    ) -> Tuple:
        """Splits a dataset into train, validation and test splits.

        Args:
            dataset (pd.DataFrame): input dataset.

        Returns:
            Tuple: tuple of train, validation and test splits. Test is None.
        """
        dataset = self.get_or_load()
        df = dataset.applymap(lambda x: torch.tensor(x))

        main_channel = list(df.columns)[0]
        aux_channels = list(df.columns)[1:]
        df_aux_all_2d = pd.DataFrame(df[aux_channels])
        df_main_all_2d = pd.DataFrame(df[main_channel])
        X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
            df_aux_all_2d, df_main_all_2d,
            test_size=self.validation_proportion, random_state=self.rnd_seed)
        return (X_train_2d, y_train_2d), (X_test_2d, y_test_2d)


class TimeSeriesProcessorSmall(DataProcessor):
    def __init__(self, name: str | None = None) -> None:
        """Preprocesses small datasets that can fit into memory.

        Args:
            name (str | None, optional): Defaults to None.
        """
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))

    @monitor_exec
    def execute(
        self,
        train_dataset: Tuple,
        validation_dataset: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Pre-process datasets: rearrange and normalize before training.

        Args:
            train_dataset (Tuple): training dataset.
            validation_dataset (Tuple): validation dataset.
            test_dataset (Any, optional): unused placeholder. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, None]: train, validation, and
                test (placeholder) datasets. Ready to be used for training.
        """
        X_train_2d, y_train_2d = train_dataset
        X_test_2d, y_test_2d = validation_dataset

        # Name of the main channel (assuming it's in position 0)
        main_channel = list(y_train_2d.columns)[0]

        # TRAINING SET

        # # smaller dataset
        # signal_data_train_small_2d = torch.stack([
        #     torch.stack([y_train_2d[main_channel].iloc[i]])
        #     for i in range(100)
        # ])  # for i in range(y_train.shape[0])
        # aux_data_train_small_2d = torch.stack([
        #     torch.stack([X_train_2d.iloc[i][0], X_train_2d.iloc[i]
        #                 [1], X_train_2d.iloc[i][2]])
        #     for i in range(100)
        # ])  # for i in range(X_train.shape[0])

        # whole dataset
        signal_data_train_2d = torch.stack([
            torch.stack([y_train_2d[main_channel].iloc[i]])
            for i in range(y_train_2d.shape[0])
        ])
        aux_data_train_2d = torch.stack([
            torch.stack(
                [X_train_2d.iloc[i][0], X_train_2d.iloc[i][1],
                 X_train_2d.iloc[i][2]])
            for i in range(X_train_2d.shape[0])
        ])

        # concatenate torch.tensors
        train_data_2d = torch.cat(
            [signal_data_train_2d, aux_data_train_2d], dim=1)
        # train_data_small_2d = torch.cat(
        #     [signal_data_train_small_2d, aux_data_train_small_2d], dim=1)

        # VALIDATION SET

        # # smaller dataset
        # signal_data_test_small_2d = torch.stack([
        #     torch.stack(
        #         [y_test_2d[main_channel].iloc[i]])
        #     for i in range(100)
        # ])  # for i in range(y_test.shape[0])
        # aux_data_test_small_2d = torch.stack([
        #     torch.stack(
        #         [X_test_2d.iloc[i][0], X_test_2d.iloc[i][1],
        #          X_test_2d.iloc[i][2]])
        #     for i in range(100)
        # ])  # for i in range(X_test.shape[0])

        # whole dataset
        signal_data_test_2d = torch.stack([
            torch.stack(
                [y_test_2d[main_channel].iloc[i]])
            for i in range(y_test_2d.shape[0])
        ])
        aux_data_test_2d = torch.stack([
            torch.stack(
                [X_test_2d.iloc[i, 0], X_test_2d.iloc[i, 1],
                 X_test_2d.iloc[i, 2]])
            for i in range(X_test_2d.shape[0])
        ])

        test_data_2d = torch.cat(
            [signal_data_test_2d, aux_data_test_2d], dim=1)
        # test_data_small_2d = torch.cat(
        #     [signal_data_test_small_2d, aux_data_test_small_2d], dim=1)

        # NORMALIZE
        train_data_2d = normalize_(train_data_2d)
        test_data_2d = normalize_(test_data_2d)

        print(f"Shape of training tensor: {train_data_2d.shape}")

        return TensorDataset(train_data_2d), TensorDataset(test_data_2d), None
