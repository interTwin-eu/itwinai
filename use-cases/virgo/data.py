import os
from typing import Optional, Tuple

import pandas as pd
import torch
from src.dataset import (
    generate_cut_image_dataset,
    generate_dataset_aux_channels,
    generate_dataset_main_channel,
    normalize_,
)
from torch.utils.data import Dataset, random_split
from trainer import NoiseGeneratorTrainer

from itwinai.components import DataGetter, DataSplitter, monitor_exec
from itwinai.pipeline import Pipeline


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
    def __init__(self, file_paths: list[str]):
        """Initialize the DataFrameDataset class.

        Args:
            file_paths (list[str]): List of paths to pickled DataFrames.
        """
        self.file_paths = file_paths

    def __len__(self):
        """Return the total number of files in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Retrieve a data sample by index, convert to tensor, and normalize.

        Args:
            idx (int): Index of the file to retrieve.

        Returns:
            torch.Tensor: Concatenated and normalized data tensor of main and auxiliary channels.
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

        # Normalize the concatenated tensor
        data_tensor = normalize_(data_tensor)

        return data_tensor


class TimeSeriesDatasetSplitter(DataSplitter):
    def __init__(
        self,
        train_proportion: int | float,
        validation_proportion: int | float = 0.0,
        test_proportion: int | float = 0.0,
        rnd_seed: int | None = None,
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

        # Ensure the root folder exists
        if not os.path.isdir(self.root_folder):
            raise FileNotFoundError(f"Root folder '{self.root_folder}' not found.")

        # Find all file paths in root folder
        all_file_paths = [os.path.join(dirpath, f)
                          for dirpath, dirs, files in os.walk(self.root_folder)
                          for f in files if f.endswith('.pkl')]

        # Split file paths into train, validation, and test sets
        [train_paths, validation_paths, test_paths] = random_split(
            all_file_paths,
            [self.train_proportion,
             self.validation_proportion,
             self.test_proportion]
        )

        # Create dataset objects for each split
        train_dataset = DataFrameDataset(train_paths)
        validation_dataset = DataFrameDataset(validation_paths)
        test_dataset = DataFrameDataset(test_paths)

        return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    root_folder = "/p/scratch/intertwin/datasets/virgo"

    pipeline = Pipeline({
        "splitter": TimeSeriesDatasetSplitter(
            train_proportion=0.5,
            validation_proportion=0.3,
            test_proportion=0.2,
            rnd_seed=42,
            root_folder=root_folder,
            name="Test Splitter"
        ),
        "trainer": NoiseGeneratorTrainer(
            generator='simple',  # unet
            batch_size=3,
            num_epochs=2,
            strategy='ddp',
            random_seed=17,
            validation_every=500,
            learning_rate=0.0005
        )
    })

    # Run pipeline
    _, _, _, trained_model = pipeline.execute()
    print("Trained model: ", trained_model)
