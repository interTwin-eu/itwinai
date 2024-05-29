from typing import Optional, Tuple, Any
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from itwinai.components import (
    DataGetter, DataProcessor, DataSplitter, monitor_exec
)

from src.dataset import (
    generate_dataset_aux_channels,
    generate_dataset_main_channel,
    generate_cut_image_dataset,
    normalize_
)


class TimeSeriesDatasetGenerator(DataGetter):
    # TODO: move configuration to the constructor.
    def __init__(
        self,
        data_root: str = "data",
        name: Optional[str] = None
    ) -> None:
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


class TimeSeriesDatasetSplitter(DataSplitter):
    def __init__(
        self,
        train_proportion: int | float,
        validation_proportion: int | float = 0.0,
        test_proportion: int | float = 0.0,
        rnd_seed: int | None = None,
        images_dataset: str = "data/Image_dataset_synthetic_64x64.pkl",
        name: str | None = None
    ) -> None:
        super().__init__(
            train_proportion, validation_proportion,
            test_proportion, name
        )
        self.save_parameters(**self.locals2params(locals()))
        self.validation_proportion = 1-train_proportion
        self.rnd_seed = rnd_seed
        self.images_dataset = images_dataset

    def get_or_load(self, dataset: Optional[pd.DataFrame] = None):
        """If the dataset is not given, load it from disk."""
        if dataset is None:
            print("WARNING: loading time series dataset from disk.")
            return pd.read_pickle(self.images_dataset)
        return dataset

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
        dataset = self.get_or_load(dataset)

        # Convert data to torch
        df = dataset.applymap(lambda x: torch.tensor(x))

        # Divide Image dataset in main and aux channels. Note that df
        # generated in the section Generate Synthetic Dataset will always have
        # the main channel as its first column
        main_channel = list(df.columns)[0]
        aux_channels = list(df.columns)[1:]

        df_aux_all_2d = pd.DataFrame(df[aux_channels])
        df_main_all_2d = pd.DataFrame(df[main_channel])
        X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
            df_aux_all_2d, df_main_all_2d,
            test_size=self.validation_proportion, random_state=self.rnd_seed)
        return (X_train_2d, y_train_2d), (X_test_2d, y_test_2d), None


class TimeSeriesProcessor(DataProcessor):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))

    @monitor_exec
    def execute(
        self,
        train_dataset: Tuple,
        validation_dataset: Tuple,
        test_dataset: Any = None
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
                [X_test_2d.iloc[i][0], X_test_2d.iloc[i][1],
                 X_test_2d.iloc[i][2]])
            for i in range(X_test_2d.shape[0])
        ])

        test_data_2d = torch.cat(
            [signal_data_test_2d, aux_data_test_2d], dim=1)
        # test_data_small_2d = torch.cat(
        #     [signal_data_test_small_2d, aux_data_test_small_2d], dim=1)

        # NORMALIZE
        train_data_2d = normalize_(train_data_2d)
        test_data_2d = normalize_(test_data_2d)

        return train_data_2d, test_data_2d, None
