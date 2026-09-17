from typing import Tuple

import torch
from torch.utils.data import Dataset, TensorDataset

from itwinai.components import DataGetter, monitor_exec

from .darcy import make_dataset


class DarcyDataGetter(DataGetter):
    """Generate Darcy flow samples by solving the PDE on random permeability fields."""

    def __init__(
        self,
        n_train: int = 256,
        n_val: int = 64,
        grid_size: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_parameters(**self.locals2params(locals()))
        self.n_train = n_train
        self.n_val = n_val
        self.grid_size = grid_size
        self.seed = seed

    @monitor_exec
    def execute(self) -> Tuple[Dataset, Dataset, None]:
        a_train, u_train = make_dataset(self.n_train, self.grid_size, self.seed)
        a_val, u_val = make_dataset(self.n_val, self.grid_size, self.seed + 1)

        mean, std = a_train.mean(), a_train.std()
        a_train, a_val = (a_train - mean) / std, (a_val - mean) / std

        return (
            TensorDataset(torch.from_numpy(a_train), torch.from_numpy(u_train)),
            TensorDataset(torch.from_numpy(a_val), torch.from_numpy(u_val)),
            None,
        )
