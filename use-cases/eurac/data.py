from typing import Optional, Tuple, Any
import os
import pandas as pd
import torch
import xarray as xr
from sklearn.model_selection import train_test_split
from itwinai.components import (
    DataGetter, DataProcessor, DataSplitter, monitor_exec
)
from hython.utils import read_from_zarr
from hython.sampler import AbstractDownSampler
from hython.normalizer import Normalizer
from hython.datasets.datasets import get_dataset

class RNNDatasetGetterAndSplitter(DataSplitter):
    def __init__(
        self,
        surrogate_input: str,
        dynamic_names: list[str],
        static_names: list[str],
        target_names: list[str],
        mask_names: list[str],
        train_temporal_range: list[str] = ["",""],
        test_temporal_range: list[str] = ["",""],
        name: str | None = None
    ) -> None:
    
        self.save_parameters(**self.locals2params(locals()))
        self.surrogate_input = surrogate_input
        self.dynamic_names = dynamic_names
        self.static_names = static_names 
        self.target_names = target_names
        self.mask_names = mask_names
        self.train_temporal_range = train_temporal_range
        self.test_temporal_range = test_temporal_range

    @monitor_exec
    def execute(
        self
    ) -> Tuple:

        # Dataset preparation
        Xd = (
            read_from_zarr(url=self.surrogate_input, group="xd", multi_index="gridcell")
            .sel(time=self.train_temporal_range)
            .xd.sel(feat=self.dynamic_names)
        )
        Xs = read_from_zarr(
            url=self.surrogate_input, group="xs", multi_index="gridcell").xs.sel(
            feat=self.static_names
        )
        Y = (
            read_from_zarr(url=self.surrogate_input, group="y", multi_index="gridcell")
            .sel(time=self.train_temporal_range)
            .y.sel(feat=self.target_names)
        )

        Y_test = (
            read_from_zarr(url=self.surrogate_input, group="y", multi_index="gridcell")
            .sel(time=self.test_temporal_range)
            .y.sel(feat=self.target_names)
        )
        Xd_test = (
            read_from_zarr(url=self.surrogate_input, group="xd", multi_index="gridcell")
            .sel(time=self.test_temporal_range)
            .xd.sel(feat=self.dynamic_names)
        )

        masks = (
            read_from_zarr(url=self.surrogate_input, group="mask")
            .mask.sel(mask_layer=self.mask_names)
            .any(dim="mask_layer")
        )
        # pass to rnnprocessor
        return (Xd, Xs, Y), (Xd_test, Xs, Y_test), None, masks


class RNNProcessor(DataProcessor):
    def __init__(self,
                 dataset: str,
                 downsampling_train: AbstractDownSampler = None, 
                 downsampling_test: AbstractDownSampler = None,
                 normalizer_dynamic: Normalizer = None,
                 normalizer_static: Normalizer = None,
                 normalizer_target: Normalizer = None, 
                 name: str | None = None) -> None:
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))
        self.dataset = dataset
        self.downsampling_train = downsampling_train
        self.downsampling_test = downsampling_test
        # For the moment these are fixed
        self.normalizer_dynamic = Normalizer(method="standardize",
                                        type="spacetime", axis_order="NTC")
                                        #save_stats=f"{TMP_STATS}/{EXPERIMENT}_xd.npy")
        self.normalizer_static = Normalizer(method="standardize",
                                        type="space", axis_order="NTC")
                                        #save_stats=f"{TMP_STATS}/{EXPERIMENT}_xs.npy")
        self.normalizer_target = Normalizer(method="standardize", type="spacetime",
                                        axis_order="NTC")
                                        #save_stats=f"{TMP_STATS}/{EXPERIMENT}_y.npy")

    @monitor_exec
    def execute(
        self,
        train_dataset: Tuple,
        validation_dataset: Tuple,
        test_dataset: Any = None,
        masks: xr.Dataset = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:

        Xd, Xs, Y = train_dataset
        Xd_test, Xs_test, Y_test = validation_dataset 

        SHAPE = Xd.attrs["shape"]
        print(SHAPE)

        train_dataset = get_dataset(self.dataset)(
                Xd,
                Y,
                Xs,
                original_domain_shape=SHAPE,
                mask=masks,
                downsampler=self.downsampling_train,
                normalizer_dynamic=self.normalizer_dynamic,
                normalizer_static=self.normalizer_static,
                normalizer_target=self.normalizer_target
        )
        validation_dataset = get_dataset(self.dataset)(
                Xd_test,
                Y_test,
                Xs_test,
                original_domain_shape=SHAPE,
                mask=masks,
                downsampler=self.downsampling_test,
                normalizer_dynamic=self.normalizer_dynamic,
                normalizer_static=self.normalizer_static,
                normalizer_target=self.normalizer_target
        )

        # Pass to trainer 
        return train_dataset, validation_dataset, None