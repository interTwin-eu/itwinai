# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Henry Mutegeki <henry.mutegeki@cern.ch> - CERN
# - Iacopo Ferrario <iacopofederico.ferrario@eurac.edu> - EURAC
# --------------------------------------------------------------------------------------

from typing import Any, Tuple

import xarray as xr
from hython.datasets import LSTMDataset, get_dataset
from hython.io import read_from_zarr
from hython.normalizer import Normalizer
from hython.sampler import AbstractDownSampler

from itwinai.components import DataProcessor, DataSplitter, monitor_exec


class RNNDatasetGetterAndSplitter(DataSplitter):
    def __init__(
        self,
        surrogate_input: str,
        dynamic_names: list[str],
        static_names: list[str],
        target_names: list[str],
        mask_names: list[str],
        train_temporal_range: list[str] = ["", ""],
        test_temporal_range: list[str] = ["", ""],
        name: str | None = None,
    ) -> None:
        self.save_parameters(**self.locals2params(locals()))
        self.surrogate_input = surrogate_input
        self.dynamic_names = dynamic_names
        self.static_names = static_names
        self.target_names = target_names
        self.mask_names = mask_names
        self.train_temporal_range = slice(train_temporal_range[0], train_temporal_range[1])
        self.test_temporal_range = slice(test_temporal_range[0], test_temporal_range[1])

    @monitor_exec
    def execute(self) -> Tuple:
        # Dataset preparation
        Xd = (
            read_from_zarr(url=self.surrogate_input, group="xd", multi_index="gridcell")
            .sel(time=self.train_temporal_range)
            .xd.sel(feat=self.dynamic_names)
        )
        Xs = read_from_zarr(
            url=self.surrogate_input, group="xs", multi_index="gridcell"
        ).xs.sel(feat=self.static_names)
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
    def __init__(
        self,
        dataset: str,
        downsampling_train: AbstractDownSampler = None,
        downsampling_test: AbstractDownSampler = None,
        normalizer_dynamic: Normalizer = None,
        normalizer_static: Normalizer = None,
        normalizer_target: Normalizer = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))
        self.dataset = dataset
        self.downsampling_train = downsampling_train
        self.downsampling_test = downsampling_test
        # For the moment these are fixed
        self.normalizer_dynamic = Normalizer(
            method="standardize", type="spacetime", axis_order="NTC"
        )
        self.normalizer_static = Normalizer(
            method="standardize", type="space", axis_order="NTC"
        )
        self.normalizer_target = Normalizer(
            method="standardize", type="spacetime", axis_order="NTC"
        )

    @monitor_exec
    def execute(
        self,
        train_dataset: Tuple,
        validation_dataset: Tuple,
        test_dataset: Any = None,
        masks: xr.Dataset = None,
    ) -> Tuple[LSTMDataset, LSTMDataset, None]:
        Xd, Xs, Y = train_dataset
        Xd_test, Xs_test, Y_test = validation_dataset

        SHAPE = Xd.attrs["shape"]

        train_dataset = get_dataset(self.dataset)(
            Xd,
            Y,
            Xs,
            original_domain_shape=SHAPE,
            mask=masks,
            downsampler=self.downsampling_train,
            normalizer_dynamic=self.normalizer_dynamic,
            normalizer_static=self.normalizer_static,
            normalizer_target=self.normalizer_target,
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
            normalizer_target=self.normalizer_target,
        )

        # Pass to trainer
        return train_dataset, validation_dataset, None


class ConvRNNDatasetGetterAndSplitter(DataSplitter):
    def __init__(
        self,
        surrogate_input: str,
        dynamic_names: list[str],
        static_names: list[str],
        target_names: list[str],
        mask_names: list[str],
        train_temporal_range: list[str] = ["", ""],
        test_temporal_range: list[str] = ["", ""],
        name: str | None = None,
    ) -> None:
        self.save_parameters(**self.locals2params(locals()))
        self.surrogate_input = surrogate_input
        self.dynamic_names = dynamic_names
        self.static_names = static_names
        self.target_names = target_names
        self.mask_names = mask_names
        self.train_temporal_range = slice(train_temporal_range[0], train_temporal_range[1])
        self.test_temporal_range = slice(test_temporal_range[0], test_temporal_range[1])

    @monitor_exec
    def execute(self) -> Tuple:
        # Dataset preparation
        Xd = read_from_zarr(url=self.surrogate_input, group="xd").sel(
            time=self.train_temporal_range
        )[self.dynamic_names]
        Xs = read_from_zarr(url=self.surrogate_input, group="xs")[self.static_names]
        Y = read_from_zarr(url=self.surrogate_input, group="y").sel(
            time=self.train_temporal_range
        )[self.target_names]

        Y_test = read_from_zarr(url=self.surrogate_input, group="y").sel(
            time=self.test_temporal_range
        )[self.target_names]

        Xd_test = read_from_zarr(url=self.surrogate_input, group="xd").sel(
            time=self.test_temporal_range
        )[self.dynamic_names]
        masks = (
            read_from_zarr(url=self.surrogate_input, group="mask")
            .mask.sel(mask_layer=self.mask_names)
            .any(dim="mask_layer")
        )
        # pass to rnnprocessor
        return (Xd, Xs, Y), (Xd_test, Xs, Y_test), None, masks


class ConvRNNProcessor(DataProcessor):
    def __init__(
        self,
        dataset: str,
        downsampling_train: AbstractDownSampler = None,
        downsampling_test: AbstractDownSampler = None,
        normalizer_dynamic: Normalizer = None,
        normalizer_static: Normalizer = None,
        normalizer_target: Normalizer = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))
        self.dataset = dataset
        self.downsampling_train = downsampling_train
        self.downsampling_test = downsampling_test
        # For the moment these are fixed
        self.normalizer_dynamic = Normalizer(
            method="standardize", type="spacetime", axis_order="xarray_dataset"
        )
        self.normalizer_static = Normalizer(
            method="standardize", type="space", axis_order="xarray_dataset"
        )
        self.normalizer_target = Normalizer(
            method="standardize", type="spacetime", axis_order="xarray_dataset"
        )

        # TODO: HARDCODED!
        XSIZE, YSIZE, TSIZE = 20, 20, 60
        XOVER, YOVER, TOVER = 5, 5, 50
        self.missing_policy = 0.05
        self.batch_size = {"xsize": XSIZE, "ysize": YSIZE, "tsize": TSIZE}
        self.overlap = {"xover": XOVER, "yover": YOVER, "tover": TOVER}
        self.persist = True
        self.static_to_dynamic = True

    @monitor_exec
    def execute(
        self,
        train_dataset: Tuple,
        validation_dataset: Tuple,
        test_dataset: Any = None,
        masks: xr.Dataset = None,
    ) -> Tuple[LSTMDataset, LSTMDataset, None]:
        Xd, Xs, Y = train_dataset
        Xd_test, Xs_test, Y_test = validation_dataset

        SHAPE = Xd.precip.shape

        train_dataset = get_dataset(self.dataset)(
            Xd,
            Y,
            Xs,
            mask=masks,
            downsampler=self.downsampling_train,
            normalizer_dynamic=self.normalizer_dynamic,
            normalizer_static=self.normalizer_static,
            normalizer_target=self.normalizer_target,
            shape=SHAPE,
            batch_size=self.batch_size,
            overlap=self.overlap,
            missing_policy=self.missing_policy,
            persist=self.persist,
            static_to_dynamic=self.static_to_dynamic,
        )
        validation_dataset = get_dataset(self.dataset)(
            Xd_test,
            Y_test,
            Xs_test,
            mask=masks,
            downsampler=self.downsampling_test,
            normalizer_dynamic=self.normalizer_dynamic,
            normalizer_static=self.normalizer_static,
            normalizer_target=self.normalizer_target,
            shape=SHAPE,
            batch_size=self.batch_size,
            overlap=self.overlap,
            missing_policy=self.missing_policy,
            persist=self.persist,
            static_to_dynamic=self.static_to_dynamic,
        )

        # Pass to trainer
        return train_dataset, validation_dataset, None
