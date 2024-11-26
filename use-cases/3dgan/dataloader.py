# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Kalliopi Tsolaki
#
# Credit:
# - Kalliopi Tsolaki <kalliopi.tsolaki@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import glob
import os
from typing import Optional

import gdown
import h5py
import lightning as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from itwinai.components import DataGetter, monitor_exec
from itwinai.loggers import Logger as BaseItwinaiLogger


class Lightning3DGANDownloader(DataGetter):
    def __init__(
        self,
        data_path: str,
        data_url: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self.save_parameters(**self.locals2params(locals()))
        super().__init__(name)
        self.data_path = data_path
        self.data_url = data_url

    @monitor_exec
    def execute(self):
        # Download data
        if not os.path.exists(self.data_path):
            if self.data_url is None:
                print("WARNING! Data URL is None. " "Skipping dataset downloading")

            gdown.download_folder(
                url=self.data_url,
                quiet=False,
                output=self.data_path,
                # verify=False
            )


class ParticlesDataset(Dataset):
    def __init__(self, datapath: str, max_samples: Optional[int] = None):
        self.datapath = datapath
        self.max_samples = max_samples
        self.data = dict()

        self.fetch_data()

    def __len__(self):
        return len(self.data["X"])

    def __getitem__(self, idx):
        return {
            "X": self.data["X"][idx],
            "Y": self.data["Y"][idx],
            "ang": self.data["ang"][idx],
            "ecal": self.data["ecal"][idx],
        }

    def fetch_data(self) -> None:
        print("Searching in :", self.datapath)
        files = sorted(glob.glob(os.path.join(self.datapath, "**/*.h5"), recursive=True))
        print("Found {} files. ".format(len(files)))
        if len(files) == 0:
            raise RuntimeError(f"No H5 files found at '{self.datapath}'!")

        # concatenated_datasets = []
        # for datafile in files:
        #     f = h5py.File(datafile, 'r')
        #     dataset = self.GetDataAngleParallel(f)
        #     concatenated_datasets.append(dataset)
        #     # Initialize result dictionary
        #     result = {key: [] for key in concatenated_datasets[0].keys()}
        #     for d in concatenated_datasets:
        #         for key in result.keys():
        #             result[key].extend(d[key])
        # return result

        for datafile in files:
            f = h5py.File(datafile, "r")
            dataset = self.GetDataAngleParallel(f)
            for field, vals_array in dataset.items():
                if self.data.get(field) is not None:
                    # Resize to include the new array
                    new_shape = list(self.data[field].shape)
                    new_shape[0] += len(vals_array)
                    self.data[field].resize(new_shape)
                    self.data[field][-len(vals_array) :] = vals_array
                else:
                    self.data[field] = vals_array

            # Stop loading data, if self.max_samples reached
            if self.max_samples is not None and len(self.data[field]) >= self.max_samples:
                for field, vals_array in self.data.items():
                    self.data[field] = vals_array[: self.max_samples]
                break

    def GetDataAngleParallel(
        self,
        dataset,
        xscale=1,
        xpower=0.85,
        yscale=100,
        angscale=1,
        angtype="theta",
        thresh=1e-4,
        daxis=-1,
    ):
        """Preprocess function for the dataset

        Args:
            dataset (str): Dataset file path
            xscale (int, optional): Value to scale the ECAL values.
                Defaults to 1.
            xpower (int, optional): Value to scale the ECAL values,
                exponentially. Defaults to 1.
            yscale (int, optional): Value to scale the energy values.
                Defaults to 100.
            angscale (int, optional): Value to scale the angle values.
                Defaults to 1.
            angtype (str, optional): Which type of angle to use.
                Defaults to "theta".
            thresh (_type_, optional): Maximum value for ECAL values.
                Defaults to 1e-4.
            daxis (int, optional): Axis to expand values. Defaults to -1.

        Returns:
          Dict: Dictionary containning the preprocessed dataset
        """
        X = np.array(dataset.get("ECAL")) * xscale
        Y = np.array(dataset.get("energy")) / yscale
        X[X < thresh] = 0
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        ecal = np.sum(X, axis=(1, 2, 3))
        indexes = np.where(ecal > 10.0)
        X = X[indexes]
        Y = Y[indexes]
        if angtype in dataset:
            ang = np.array(dataset.get(angtype))[indexes]
        # else:
        # ang = gan.measPython(X)
        X = np.expand_dims(X, axis=daxis)
        ecal = ecal[indexes]
        ecal = np.expand_dims(ecal, axis=daxis)
        if xpower != 1.0:
            X = np.power(X, xpower)

        Y = np.array([[el] for el in Y])
        ang = np.array([[el] for el in ang])
        ecal = np.array([[el] for el in ecal])

        final_dataset = {"X": X, "Y": Y, "ang": ang, "ecal": ecal}

        return final_dataset


class ParticlesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datapath: str,
        batch_size: int,
        num_workers: int = 4,
        max_samples: Optional[int] = None,
        train_proportion: float = 0.9,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datapath = datapath
        self.max_samples = max_samples
        self.train_proportion = train_proportion

    @property
    def itwinai_logger(self) -> BaseItwinaiLogger:
        try:
            itwinai_logger = self.trainer.itwinai_logger
        except AttributeError:
            print("WARNING: itwinai_logger attribute not set " f"in {self.__class__.__name__}")
            itwinai_logger = None
        return itwinai_logger

    def setup(self, stage: str = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        if stage == "fit" or stage is None:
            self.dataset = ParticlesDataset(self.datapath, max_samples=self.max_samples)
            dataset_length = len(self.dataset)
            split_point = int(dataset_length * self.train_proportion)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset, [split_point, dataset_length - split_point]
            )

        if stage == "predict":
            # TODO: inference dataset should be different in that it
            # does not contain images!
            self.predict_dataset = ParticlesDataset(
                self.datapath, max_samples=self.max_samples
            )

        # if stage == 'test' or stage is None:
        # self.test_dataset = MyDataset(self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
        )

    # def test_dataloader(self):
    # return DataLoader(self.test_dataset, batch_size=self.batch_size)
