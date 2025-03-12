import os
from pathlib import Path
from timeit import default_timer
from typing import Dict, Literal, Optional, Union, Any, Tuple
from tqdm.auto import tqdm
import copy
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pandas as pd
from ray import train

import xarray as xr
from hython.scaler import Scaler
from hython.sampler import SamplerBuilder
from hython.models import get_model_class as get_hython_model
from hython.models import load_model, ModelLogAPI
from itwinai.components import monitor_exec
from hython.utils import prepare_for_plotting2d


from itwinai.torch.distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    NonDistributedStrategy,
    TorchDDPStrategy,
)

from itwinai.torch.monitoring.monitoring import measure_gpu_utilization
from itwinai.distributed import suppress_workers_print
from itwinai.loggers import EpochTimeTracker, Logger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.inference import TorchPredictor, ModelLoader 
from itwinai.components import Predictor
from itwinai.torch.type import Metric
from itwinai.torch.profiling.profiler import profile_torch_trainer


def create_xarray_dataset(
    y_target,
    shape,
    coords,
    dim_variable_name = "variable",
    crs = 4326
):

    lat, lon = shape
    
    n_feat = y_target.shape[-1]

    y = y_target.reshape(lat, lon, n_feat)

    ds = xr.DataArray(y, dims=["lat", "lon", "variable"], coords=coords).to_dataset(dim=dim_variable_name)

    if crs:
        ds.rio.write_crs(4236)

    return ds

class ParameterInference(Predictor):

    def __init__(self,
                 test_dataset: Dataset | None = None, 
                 test_dataloader: DataLoader | None = None, 
                 model: Union[nn.Module, ModelLoader, None] = None,
                 scaling_static_range: Dict | None = None):
        super().__init__(model = model)
        self.save_parameters(**self.locals2params(locals()))
        self.scaling_static_range = scaling_static_range

    @monitor_exec
    def execute(
        self,
        test_dataset: Dataset,
        dataloader: DataLoader,
        model: nn.Module = None,
        strategy = None, 
        cfg = None
    ) -> Dict[str, Any]:

        if model is not None:
            # Overrides existing "internal" model
            self.model = model
        transfer_nn = model.transfernn

        device = strategy.device()

        self.scaling_static_range = {k:self.scaling_static_range[k] for k in cfg.head_model_inputs}

        scaler = Scaler(cfg)

        params = []
        transfer_nn.eval()
        for data in dataloader:
            xs = data["xs"]
            out = transfer_nn(xs.to(device))
            params.append(out.detach())

        params = torch.concat(params, 0).detach()

        scale = torch.tensor([range[1] - range[0] for v, range in self.scaling_static_range.items() ]).to(device)
        center = torch.tensor([range[0] for v, range in self.scaling_static_range.items() ]).to(device)  
        
        params_orig = scaler.transform_inverse_custom_range(params, scale=scale, center= center)

        coords = xr.Coordinates({"lat":test_dataset.xd.lat, "lon":test_dataset.xd.lon, "variable":cfg.head_model_inputs})

        lat, lon = len(test_dataset.xs.lat),len(test_dataset.xs.lon)

        y_params = prepare_for_plotting2d(
                                        y_target= params_orig.cpu(),
                                        shape = (lat, lon),
                                        coords  = coords)
        
        y_params = y_params.to_dataset("variable")

        y_params.to_netcdf(f"{cfg.work_dir}/test.nc")


