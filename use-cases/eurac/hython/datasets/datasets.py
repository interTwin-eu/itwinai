import torch
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset


class LSTMDataset(Dataset):

    def __init__(self, 
                 xd:torch.Tensor | npt.NDArray, 
                 y:torch.Tensor  | npt.NDArray, 
                 xs:torch.Tensor | npt.NDArray = None, 
                 seq_length:int = 60, 
                 create_seq:bool = False):
        """Create a dataset for training and validating LSTM-based models

        Args:
            xd (torch.Tensor | npt.NDArray): dynamic input. Expected shape: (LOCATION, TIME, FEATURE). 
                In case data is from a gridded (raster) dataset, location means gridcell.
            y (torch.Tensor | npt.NDArray): target (LOCATION, TIME, FEATURE)
            xs (torch.Tensor | npt.NDArray, optional): static input (LOCATION, FEATURE). Defaults to None.
            seq_length (int, optional): length of the time series sequences. Defaults to 60.
            create_seq (bool, optional): alternative way of reshaping data. Defaults to False.
        """
        self.seq_len = seq_length
        self.create_seq = create_seq

        num_gridcells, num_samples, num_features = xd.shape

        if isinstance(xs, np.ndarray):
            self.xs = xs.astype(np.float32)
        else:
            self.xs = xs

        if create_seq:
    
            xd_new = np.zeros((num_gridcells, num_samples - seq_length + 1, seq_length, num_features))
            y_new = np.zeros((num_gridcells, num_samples - seq_length + 1, 1))
        
            for i in range(0, xd_new.shape[1]):
                xd_new[:, i, :, :num_features] = xd[:, i:i + seq_length, :]
                y_new[:, i, 0] = y[:, i + seq_length - 1, 0]
            
            self.Xd = torch.from_numpy(xd_new.astype(np.float32))
            self.y =  torch.from_numpy(y_new.astype(np.float32))
        else:

            self.Xd = xd 
            self.y = y 
        
    def __len__(self):
        return self.Xd.shape[0]

    def __getitem__(self, i):
        if self.xs is not None:
            if self.create_seq:
                return self.Xd[i], self.xs, self.y[i]
            else:
                return self.Xd[i], self.xs[i], self.y[i]
        else:
            return self.Xd[i], self.y[i]