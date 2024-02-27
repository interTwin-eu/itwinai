from itwinai.components import DataGetter, monitor_exec
from typing import List, Dict

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset

#import gdown
import numpy as np
import xarray as xr
#import hvplot.xarray 
from preprocess import preprocess, apply_normalization
from utils import missing_location_idx
from datasets import LSTMDataset
from sampler import RegularIntervalSampler

class LSTMDataGetter(DataGetter):
    def __init__(
        self,
        wd: str, 
        fp_dynamic_forcings: str, 
        fp_wflow_static_params: str, 
        fp_target: str, 
        dynamic_names: list, 
        static_names : list, 
        target_names : list, 
        timeslice    : list, 
        intervals    : list,
        train_origin : list, 
        valid_origin : list, 
        spatial_batch_size: int = None, 
        temporal_sampling_size: int = None, 
        seq_length : int = None, 
        hidden_size: int = None, 
        input_size : int = None, #number of dynamic predictors - user_input
    ):
        super().__init__()
        self.save_parameters(**self.locals2params(locals()))

        self.wd = wd
        self.fp_dynamic_forcings    = fp_dynamic_forcings
        self.fp_wflow_static_params = fp_wflow_static_params
        self.fp_target     = fp_target
        self.dynamic_names =  dynamic_names
        self.static_names  =  static_names 
        self.target_names  =  target_names 
        self.timeslice     =  slice(timeslice[0], timeslice[1])
        self.intervals     = tuple(intervals)
        self.train_origin  = tuple(train_origin)
        self.valid_origin  = tuple(valid_origin)
        self.spatial_batch_size = spatial_batch_size
        self.temporal_sampling_size = temporal_sampling_size
        self.seq_length    = seq_length
        self.hidden_size   = hidden_size 
        self.input_size    = input_size

        self.model_params={
            "input_size": 3, #number of dynamic predictors - user_input
            "hidden_size": hidden_size, # user_input
            "output_size": len(target_names), # number_target - user_input
            "number_static_predictors": len(static_names), #number of static parameters - user_input 
        }

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def normalization(self, Xd, Xs, Y, d_m: float | None = None, s_m : float | None = None, d_std: float | None = None, s_std : float | None = None):
            Xd, d_m, d_std = apply_normalization(Xd, type = "spacetime", how ='standard', m1 = d_m, m2 = d_std )
            Xs, s_m, s_std = apply_normalization(Xs, type = "space", how ='standard', m1 = s_m, m2 = s_std)
            #Y_clean, y_m, y_std = apply_normalization(Y_clean, type = "spacetime", how ='standard')
        
            # to tensor
            Xs, Xd = torch.Tensor(Xs), torch.Tensor(Xd)
            Y = torch.Tensor(Y)
            return Xd, Xs, Y, d_m, d_std, s_m, s_std

    @monitor_exec
    def execute(self):
        
        forcings = xr.open_dataset( self.fp_dynamic_forcings).rename({"latitude":"lat", "longitude":"lon"})
        params = xr.open_dataset(self.fp_wflow_static_params ).rename({"latitude":"lat", "longitude":"lon"})
        targets = xr.open_dataset(self.fp_target).isel(lat=slice(None, None, -1))

        if self.timeslice:
            forcings = forcings.sel(time=self.timeslice)
            targets = targets.sel(time=self.timeslice)

        #this does not belong here. 
        #params.wflow_dem.hvplot(geo=True, tiles=True, cmap = "terrain").opts(width = 799, height= 600)
        
        #define data mask
        mask_lakes = (targets.mean(dim = "time")["actevap"] == 0).astype(np.bool_)
        
        Xd, Xs, Y, dims, meta = preprocess(forcings, 
                   params, 
                   targets, 
                   dynamic_name = self.dynamic_names,
                   static_name = self.static_names, 
                   target_name = self.target_names, 
                   sampler = RegularIntervalSampler(
                                 intervals = self.intervals,
                                 origin = self.train_origin
                    )
                   )
        Xd_valid, Xs_valid, Y_valid, dims_valid, meta_valid = preprocess(
                    forcings, 
                    params, 
                    targets,
                    dynamic_name = self.dynamic_names,
                    static_name = self.static_names, 
                    target_name = self.target_names, 
                    sampler = RegularIntervalSampler(
                            intervals = self.intervals,
                            origin = self.valid_origin)
                    )

        # Remove missing values
        # Find indices of missing values
        idx = missing_location_idx(Xs)
        idx_valid = missing_location_idx(Xs_valid)

        Xd_clean = Xd[~idx]
        Xs_clean = Xs[~idx]
        Y_clean = Y[~idx]
        Xd_valid_clean = Xd_valid[~idx_valid]
        Xs_valid_clean = Xs_valid[~idx_valid]
        Y_valid_clean = Y_valid[~idx_valid]
        
        Xd_clean, Xs_clean, Y_clean, d_m, d_std, s_m, s_std  = self.normalization(Xd_clean, Xs_clean, Y_clean)
        Xd_clean, Xs_valid_clean, Y_valid_clean, d_m, d_std, s_m, s_std = self.normalization(Xd_valid_clean, Xs_valid_clean, Y_valid_clean, d_m, s_m, d_std, s_std)
        
        train_dataset = LSTMDataset(Xd_clean, Y_clean, Xs_clean)
        val_dataset = LSTMDataset(Xd_valid_clean, Y_valid_clean, Xs_valid_clean)

        train_loader = DataLoader(train_dataset, batch_size=self.spatial_batch_size, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=self.spatial_batch_size, shuffle=False)