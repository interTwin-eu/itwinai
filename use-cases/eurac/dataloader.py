from itwinai.components import DataGetter, monitor_exec
from typing import List, Dict

import torch

#import gdown
import numpy as np
import xarray as xr
#import hvplot.xarray 
from hython.preprocess import preprocess, apply_normalization
from hython.utils import missing_location_idx
from hython.datasets.datasets import LSTMDataset
from hython.sampler import RegularIntervalSampler

from itwinai.torch.distributed import (
    TorchDistributedStrategy,
    DDPDistributedStrategy,
    HVDDistributedStrategy,
   # DSDistributedStrategy,
)


class LSTMDataGetter(DataGetter):
    def __init__(
        self,
        wd: str, 
        fp_dynamic_forcings: str, 
        fp_wflow_static_params: str, 
        fp_target: str, 
        surrogate_data: str, 
        dp_surrogate_model: str,
        strategy_name : str,
        dynamic_names : list, 
        static_names  : list, 
        target_names  : list, 
        timeslice     : list, 
        intervals     : list,
        train_origin  : list, 
        valid_origin  : list, 
        spatial_batch_size: int = None, 
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
        # self.shuffle_dataloader = shuffle_dataloader
  
        # Instantiate Strategy
        self.is_distributed = True
        if strategy_name == 'ddp':
            if (not torch.cuda.is_available()
                    or not torch.cuda.device_count() > 1):
                raise RuntimeError('Resources unavailable')
            self.strategy = DDPDistributedStrategy(backend='nccl')
        elif strategy_name == 'horovod':
            self.strategy = HVDDistributedStrategy()
        elif strategy_name == 'deepspeed':
            self.strategy = DSDistributedStrategy(backend='nccl')
        elif strategy_name == 'None':
            self.strategy = None
            self.is_distributed = False
        else:
            raise NotImplementedError(
                f"Strategy {strategy} is not recognized/implemented.")

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
        if self.is_distributed:
            self.strategy.init()
            print(f"{foo}: {os.environ.get('RANK')} {os.environ.get('LOCAL_RANK')} "
              f"{os.environ.get('MASTER_ADDR')} {os.environ.get('MASTER_PORT')}")
         
        Xd = read_from_zarr(url=surrogate_data, group="xd", multi_index="gridcell").sel(time = train_range).xd
        Xs = read_from_zarr(url=surrogate_data, group="xs", multi_index="gridcell").xs
        Y = read_from_zarr(url=surrogate_data, group="y", multi_index="gridcell").sel(time = train_range).y

        # other 
        # wflow_lakes = Xs.sel(feat="wflow_lakeareas").unstack()
        wflow_dem = Xs.sel(feat="wflow_dem").unstack()

        # select features and targets 
        Xd = Xd.sel(feat=dynamic_names)
        Xs = Xs.sel(feat=static_names)
        Y = Y.sel(feat=target_names)

        print(Xd.shape, Xs.shape, Y.shape)
        # read masks
        mask_missing = read_from_zarr(url=surrogate_data, group="mask" ).mask
        # mask_lake = read_from_zarr(url=surrogate_data, group="mask_lake" ).mask_lake
        
        spatial_train_sampler = RegularIntervalSampler(intervals = INTERVALS, origin = TRAIN_ORIGIN)
        spatial_val_sampler = RegularIntervalSampler(intervals = INTERVALS, origin = VAL_ORIGIN) 

        # Apply the samplers: return the cell indices that can be used later in training and validation to sample the whole spatial domain.
        data2d  = wflow_dem.values

        idx = missing_location_idx(Xs.values)

        sampler_train_meta = spatial_train_sampler.sampling_idx(data2d, mask_missing)
        sampler_val_meta = spatial_val_sampler.sampling_idx(data2d, mask_missing)

        # some useful metadata
        print(sampler_train_meta)
        Xd_clean = Xd[sampler_train_meta.idx_sampled_1d_nomissing]
        Xs_clean = Xs[sampler_train_meta.idx_sampled_1d_nomissing]
        Y_clean  = Y[sampler_train_meta.idx_sampled_1d_nomissing]

        _, _, _, d_m, d_std, s_m, s_std  = self.normalization(Xd_clean, Xs_clean, Y_clean)
        Xd, Xs, Y, d_m, d_std, s_m, s_std = self.normalization(Xd, Xs, Y, d_m, s_m, d_std, s_std)
        
        # init datasets
        dataset = LSTMDataset(Xd, Y, Xs)

        train_sampler = DataLoaderSpatialSampler(dataset, num_samples=1, sampling_indices = sampler_train_meta.idx_sampled_1d_nomissing.tolist())
        valid_sampler = DataLoaderSpatialSampler(dataset, num_samples=1, sampling_indices = sampler_val_meta.idx_sampled_1d_nomissing.tolist())

        return dataset, train_sampler, valid_sampler