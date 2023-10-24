from typing import Optional, Tuple, Dict
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as pl
import glob
import h5py
import gdown

from itwinai.components import DataGetter


class Lightning3DGANDownloader(DataGetter):
    def __init__(
            self,
            data_url: str,
            data_path: str,
            name: Optional[str] = None,
            **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.data_path = data_path
        self.data_url = data_url

    def load(self):
        # Download data
        if not os.path.exists(self.data_path):
            gdown.download_folder(
                url=self.data_url, quiet=False,
                output=self.data_path
            )

    def execute(
        self,
        config: Optional[Dict] = None
    ) -> Tuple[None, Optional[Dict]]:
        self.load()
        return None, config


class MyDataset(Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        self.data = self.fetch_data(self.datapath)

    def __len__(self):
        return len(self.data["X"])

    def __getitem__(self, idx):
        return {"X": self.data["X"][idx], "Y": self.data["Y"][idx], "ang": self.data["ang"][idx], "ecal": self.data["ecal"][idx]}

    def fetch_data(self, datapath):

        print("Searching in :", datapath)
        Files = sorted(glob.glob(datapath))
        print("Found {} files. ".format(len(Files)))

        concatenated_datasets = []
        for datafile in Files:
            f = h5py.File(datafile, 'r')
            dataset = self.GetDataAngleParallel(f)
            concatenated_datasets.append(dataset)
            # Initialize result dictionary
            result = {key: [] for key in concatenated_datasets[0].keys()}
            for d in concatenated_datasets:
                for key in result.keys():
                    result[key].extend(d[key])
        return result

    def GetDataAngleParallel(
            self,
            dataset,
            xscale=1,
            xpower=0.85,
            yscale=100,
            angscale=1,
            angtype="theta",
            thresh=1e-4,
            daxis=-1,):
        """Preprocess function for the dataset

        Args:
            dataset (str): Dataset file path
            xscale (int, optional): Value to scale the ECAL values. Defaults to 1.
            xpower (int, optional): Value to scale the ECAL values, exponentially. Defaults to 1.
            yscale (int, optional): Value to scale the energy values. Defaults to 100.
            angscale (int, optional): Value to scale the angle values. Defaults to 1.
            angtype (str, optional): Which type of angle to use. Defaults to "theta".
            thresh (_type_, optional): Maximum value for ECAL values. Defaults to 1e-4.
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


class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, datapath):
        super().__init__()
        self.batch_size = batch_size
        self.datapath = datapath

    def setup(self, stage: str = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        if stage == 'fit' or stage is None:
            self.dataset = MyDataset(self.datapath)
            dataset_length = len(self.dataset)
            split_point = int(dataset_length * 0.9)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset, [split_point, dataset_length - split_point])

        # if stage == 'test' or stage is None:
            # self.test_dataset = MyDataset(self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True)

    # def test_dataloader(self):
        # return DataLoader(self.test_dataset, batch_size=self.batch_size)
