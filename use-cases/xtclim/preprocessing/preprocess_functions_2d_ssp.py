#!/usr/bin/env python
# coding: utf-8

# ## Preprocess Data for VAE

# The aim of this notebook is to translate NetCDF files (.nc)
# of a daily climate variable (e.g. maximum temperature)
# to a numpy 3D-array. This output array can be read for training
# and evaluating the Convolutional Variational AutoEncoder model.


# #### 0. Libraries

import numpy as np
import xarray as xr
import cftime
import pandas as pd

from itwinai.components import DataGetter, monitor_exec

class PreprocessData(DataGetter):
    def __init__(
            self,
            scenario: str,
            dataset_root: str
    ):
        super().__init__()
        self.scenario = scenario
        self.dataset_root = dataset_root

    def xr_to_ndarray(self, xr_dset: xr.Dataset, sq_coords: dict) -> (np.ndarray, np.array, str):
        """
        Converts an xarray dataset it to a cropped square ndarray,
        after ajusting the longitudes from [0,360] to [-180,180].

        Parameters:
        xr_dset: data set of a climate variable
        sq_coords: spatial coordinates of the crop
        """
        # adjust the longitudes to keep a continuum over Europe
        xr_dset.coords["lon"] = (xr_dset.coords["lon"] + 180) % 360 - 180
        # re-organize data
        xr_dset = xr_dset.sortby(xr_dset.lon)
        # crop to the right square
        xr_dset = xr_dset.sel(
            lon=slice(sq_coords["min_lon"], sq_coords["max_lon"]),
            lat=slice(sq_coords["min_lat"], sq_coords["max_lat"]),
        )
        time_list = np.array(xr_dset["time"])
        n_t = len(time_list)
        n_lat = len(xr_dset.coords["lat"])
        n_lon = len(xr_dset.coords["lon"])
        nd_dset = np.ndarray((n_t, n_lat, n_lon, 1), dtype="float32")
        climate_variable = xr_dset.attrs["variable_id"]
        nd_dset[:, :, :, 0] = xr_dset[climate_variable][:, :, :]
        nd_dset = np.flip(nd_dset, axis=1)

        return nd_dset, time_list

    def sftlf_to_ndarray(
        self, xr_dset: xr.Dataset, sq_coords: dict
    ) -> (np.ndarray, np.array, str):
        """
        Converts and normalizes the land-sea mask data set
        to a cropped square ndarray,
        after ajusting the longitudes from [0,360] to [-180,180].

        Parameters:
        xr_dset: land-sea mask data set
        sq_coords: spatial coordinates of the crop
        """
        xr_dset.coords["lon"] = (xr_dset.coords["lon"] + 180) % 360 - 180
        xr_dset = xr_dset.sortby(xr_dset.lon)
        xr_dset = xr_dset.sel(
            lon=slice(sq_coords["min_lon"], sq_coords["max_lon"]),
            lat=slice(sq_coords["min_lat"], sq_coords["max_lat"]),
        )
        lat_list = xr_dset.coords["lat"]
        lon_list = xr_dset.coords["lon"]
        n_lat = len(lat_list)
        n_lon = len(lon_list)
        land_prop = np.ndarray((n_lat, n_lon, 1), dtype="float32")
        climate_variable = xr_dset.attrs["variable_id"]
        land_prop[:, :, 0] = xr_dset[climate_variable][:, :]
        # flip upside down to have North up
        land_prop = np.flipud(land_prop)
        # normalize data (originally in [0,100])
        land_prop = land_prop / 100

        return land_prop, lat_list, lon_list

    def get_extrema(self, histo_dataset: np.ndarray, proj_dataset: np.ndarray) -> np.array:
        """
        Computes global extrema over past and future data.

        Parameters:
        histo_dataset: historical data
        proj_dataset: projection data
        """
        global_min = min(np.min(histo_dataset), np.min(proj_dataset))
        global_max = max(np.max(histo_dataset), np.max(proj_dataset))
        return np.array([global_min, global_max])

    def normalize(self, nd_dset: np.ndarray, extrema: np.array) -> np.ndarray:
        """
        Normalizes a data set with given extrema.

        Parameters:
        nd_dset: data set of a climate variable
        extrema: extrema of the climate variable ([min, max])
        """
        norm_dset = (nd_dset - extrema[0]) / (extrema[1] - extrema[0])
        return norm_dset

    ##### 4. Split Historical Data into Train and Test Datasets
    # Train the network on most of the historical data,
    # but keep some to test the model performance on new data points.
    def split_train_test(
        self, nd_dset: np.ndarray, time_list: np.array, train_proportion: float = 0.8
    ) -> (np.ndarray, np.ndarray, np.array, np.array):
        """
        Splits a data set into train and test data (and time).

        Parameters:
        nd_dset: data set to be split
        time_list: time list to be split
        train_proportion: proportion of train data
        """
        len_train = int(len(nd_dset) * train_proportion)
        train_data = nd_dset[:len_train]
        test_data = nd_dset[len_train:]
        train_time = time_list[:len_train]
        test_time = time_list[len_train:]
        return train_data, test_data, train_time, test_time

    def split_date(
        self, nd_dset: np.ndarray, time_list: np.array, year: int = 1950
    ) -> (np.ndarray, np.ndarray, np.array, np.array):
        """
        Splits a data set into train and test data (and time),
        if the previous train_proportion splits data in the middle of a year.

        Parameters:
        nd_dset: data set to be split
        time_list: time list to be split
        year: year where the data is to be split
        """
        split_index = np.where(
            time_list == cftime.DatetimeNoLeap(year, 1, 1, 0, 0, 0, 0, has_year_zero=True)
        )[0][0]
        train_data = nd_dset[:split_index]
        test_data = nd_dset[split_index:]
        train_time = time_list[:split_index]
        test_time = time_list[split_index:]
        return train_data, test_data, train_time, test_time

    ##### 5. Combine into a 2D-Array
    def ndarray_to_2d(self, temp_dset: np.ndarray, land_prop: np.ndarray) -> np.ndarray:
        """
        Combines climate variable and land-sea mask data sets.

        Parameters:
        temp_dset: climate variable data
        land_prop: land-sea mask data
        """
        n_t = np.shape(temp_dset)[0]
        n_lat = np.shape(temp_dset)[1]
        n_lon = np.shape(temp_dset)[2]

        # combine all variables on a same period to a new 2D-array
        total_dset = np.zeros((n_t, n_lat, n_lon, 2), dtype="float32")
        total_dset[:, :, :, 0] = temp_dset.reshape(n_t, n_lat, n_lon)
        total_dset[:, :, :, 1] = np.transpose(
            np.repeat(land_prop, n_t, axis=2), axes=[2, 0, 1]
        )

        return total_dset

    @monitor_exec
    def execute(self):
        #### 1. Load Data to xarrays
        data_dir = self.dataset_root

        # Historical Datasets
        # regrouped by climate variable

        temp_50 = xr.open_dataset(
            f"{data_dir}tasmax_day_CMCC-ESM2_historical_r1i1p1f1_gn_19500101-19741231.nc"
        )
        temp_75 = xr.open_dataset(
            f"{data_dir}tasmax_day_CMCC-ESM2_historical_r1i1p1f1_gn_19750101-19991231.nc"
        )
        temp_00 = xr.open_dataset(
            f"{data_dir}tasmax_day_CMCC-ESM2_historical_r1i1p1f1_gn_20000101-20141231.nc"
        )
        temp_histo = xr.concat([temp_50, temp_75, temp_00], "time")

        # Projection Datasets
        # regrouped by climate variable
        # IPCC scenarios: SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5
        # choose among "126", "245", "370", "585"
        scenario = self.scenario

        temp_15 = xr.open_dataset(
            f"{data_dir}tasmax_day_CMCC-ESM2_ssp{scenario}_r1i1p1f1_gn_20150101-20391231.nc"
        )
        temp_40 = xr.open_dataset(
            f"{data_dir}tasmax_day_CMCC-ESM2_ssp{scenario}_r1i1p1f1_gn_20400101-20641231.nc"
        )
        temp_65 = xr.open_dataset(
            f"{data_dir}tasmax_day_CMCC-ESM2_ssp{scenario}_r1i1p1f1_gn_20650101-20891231.nc"
        )
        temp_90 = xr.open_dataset(
            f"{data_dir}tasmax_day_CMCC-ESM2_ssp{scenario}_r1i1p1f1_gn_20900101-21001231.nc"
        )
        temp_proj = xr.concat([temp_15, temp_40, temp_65, temp_90], "time")

        # Load land-sea mask data
        sftlf = xr.open_dataset(
            f"{data_dir}sftlf_fx_CESM2_historical_r9i1p1f1_gn.nc",
            chunks={"time": 10},
        )

        ##### 2. Restrict to a Geospatial Square
        sq32_west_europe = {"min_lon": -10, "max_lon": 29, "min_lat": 36, "max_lat": 66}

        land_prop, lat_list, lon_list = self.sftlf_to_ndarray(sftlf, sq32_west_europe)

        ##### 6. Step-by-Step Preprocessing

        # Crop original data to Western Europe
        temp_histo_nd, time_list = self.xr_to_ndarray(temp_histo, sq32_west_europe)
        temp_proj_nd, time_proj = self.xr_to_ndarray(temp_proj, sq32_west_europe)


        # Compute the variable extrema over history and projections
        # temp_extrema = get_extrema(temp_histo_nd, temp_proj_nd)

        # Here are the results for CMCC-ESM2 (all scenarios)
        # to save time
        temp_extrema = np.array([234.8754, 327.64])
        # ssp585 array([234.8754, 327.64  ], dtype=float32)
        # ssp370 array([234.8754 , 325.43323], dtype=float32)
        # ssp245 array([234.8754, 324.8263], dtype=float32)
        # ssp126 array([234.8754, 323.6651], dtype=float32)

        # Normalize data with the above extrema
        temp_histo_norm = self.normalize(temp_histo_nd, temp_extrema)
        temp_proj_norm = self.normalize(temp_proj_nd, temp_extrema)

        # Split data into train and test data sets
        train_temp, test_temp, train_time, test_time = self.split_train_test(
            temp_histo_norm, time_list
        )

        # Add up split data and land-sea mask
        total_train = self.ndarray_to_2d(train_temp, land_prop)
        total_test = self.ndarray_to_2d(test_temp, land_prop)
        total_proj = self.ndarray_to_2d(temp_proj_norm, land_prop)


        ##### 7. Save Results

        # Save train and test data sets
        np.save("input/preprocessed_2d_train_data_allssp.npy", total_train)
        np.save("input/preprocessed_2d_test_data_allssp.npy", total_test)
        pd.DataFrame(train_time).to_csv("input/dates_train_data.csv")
        pd.DataFrame(test_time).to_csv("input/dates_test_data.csv")

        # Save projection data for one scenario
        np.save(f"input/preprocessed_2d_proj{scenario}_data_allssp.npy", total_proj)
        pd.DataFrame(time_proj).to_csv("input/dates_proj_data.csv")


        ##### 8. Preprocessing for All Scenarios

        # This part is to be run as a complement to 6. and 7.

        # Here you can remove the scenario you already run in 6. and 7.
        #scenarios = ["126", "245", "370", "585"]
        #TODO: Discuss with Anne/Christian
        #scenarios = ["245", "585"]
        scenarios = [self.scenario]

        for scenario in scenarios:

            # Load projection data
            temp_15 = xr.open_dataset(
                f"{data_dir}tasmax_day_CMCC-ESM2_ssp{scenario}_r1i1p1f1_gn_20150101-20391231.nc"
            )
            temp_40 = xr.open_dataset(
                f"{data_dir}tasmax_day_CMCC-ESM2_ssp{scenario}_r1i1p1f1_gn_20400101-20641231.nc"
            )
            temp_65 = xr.open_dataset(
                f"{data_dir}tasmax_day_CMCC-ESM2_ssp{scenario}_r1i1p1f1_gn_20650101-20891231.nc"
            )
            temp_90 = xr.open_dataset(
                f"{data_dir}tasmax_day_CMCC-ESM2_ssp{scenario}_r1i1p1f1_gn_20900101-21001231.nc"
            )
            temp_proj = xr.concat([temp_15, temp_40, temp_65, temp_90], "time")

            # Process projection data
            temp_proj_nd, time_proj = self.xr_to_ndarray(temp_proj, sq32_west_europe)
            # extrema for all scenarios in CMCC-ESM2
            temp_extrema = np.array([234.8754, 327.64])
            temp_proj_norm = self.normalize(temp_proj_nd, temp_extrema)
            total_proj = self.ndarray_to_2d(temp_proj_norm, land_prop)

            # Save results
            np.save(f"input/preprocessed_2d_proj{scenario}_data_allssp.npy", total_proj)
