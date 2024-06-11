#!/usr/bin/env python
# coding: utf-8

# ## Preprocess Data for seasonal VAEs

# The aim of this notebook is to translate numpy files (.npy) and time series (.csv)
# of daily maximum temperature to four numpy 3D-arrays: one for each season.
# These output arrays can easily be read for training and evaluating the
# Convolutional Variational AutoEncoder model.

# #### 0. Libraries

import numpy as np
import pandas as pd


# #### 1. Load Data to xarray

# choose the needed number of members
n_memb = 1

# define relevant scenarios
#scenarios = ["126", "245", "370", "585"]
scenarios = ["245", "585"]

# Load preprocessed "daily temperature images" and time series

train_images = np.load("../input/preprocessed_2d_train_data_allssp.npy")
test_images = np.load("../input/preprocessed_2d_test_data_allssp.npy")
train_time = pd.read_csv("../input/dates_train_data.csv")
test_time = pd.read_csv("../input/dates_test_data.csv")


# #### 2. Split Yearly Data into Four Seasonal Datasets

# split daily data into seasons


def season_split(
    images: np.ndarray,
    time: pd.DataFrame,
    dataset_type: str,
    n_memb: int,
    scenario: str = "",
) -> tuple[list[np.ndarray], list[pd.DataFrame]]:
    """
    Splits and returns the data sets (climate variable and time) per season.

    Parameters:
    images: temperature data sets
    time: time data sets
    dataset_type: 'train', 'test', 'proj'
    n_memb: number of ensemble members
    scenario: '' for train or test data,
              or '126', '245', '370', '585' for projections
    """

    n_years = int(len(images) / 365)

    # 1st April = index 90
    # 1st July = index 181
    # 1st October = index 273
    winter_index = [365 * i + j for i in range(n_years) for j in range(90)]
    spring_index = [365 * i + j for i in range(n_years) for j in range(90, 181)]
    summer_index = [365 * i + j for i in range(n_years) for j in range(181, 273)]
    autumn_index = [365 * i + j for i in range(n_years) for j in range(273, 365)]

    winter_images = images[winter_index]
    spring_images = images[spring_index]
    summer_images = images[summer_index]
    autumn_images = images[autumn_index]

    winter_time = time.loc[winter_index].iloc[:, 1]
    spring_time = time.loc[spring_index].iloc[:, 1]
    summer_time = time.loc[summer_index].iloc[:, 1]
    autumn_time = time.loc[autumn_index].iloc[:, 1]

    # save results as an input for CVAE training
    np.save(
        f"../input/preprocessed_1d_{dataset_type}{scenario}_winter_data_{n_memb}memb.npy",
        winter_images,
    )
    np.save(
        f"../input/preprocessed_1d_{dataset_type}{scenario}_spring_data_{n_memb}memb.npy",
        spring_images,
    )
    np.save(
        f"../input/preprocessed_1d_{dataset_type}{scenario}_summer_data_{n_memb}memb.npy",
        summer_images,
    )
    np.save(
        f"../input/preprocessed_1d_{dataset_type}{scenario}_autumn_data_{n_memb}memb.npy",
        autumn_images,
    )
    pd.DataFrame(winter_time).to_csv(
        f"../input/dates_{dataset_type}_winter_data_{n_memb}memb.csv"
    )
    pd.DataFrame(spring_time).to_csv(
        f"../input/dates_{dataset_type}_spring_data_{n_memb}memb.csv"
    )
    pd.DataFrame(summer_time).to_csv(
        f"../input/dates_{dataset_type}_summer_data_{n_memb}memb.csv"
    )
    pd.DataFrame(autumn_time).to_csv(
        f"../input/dates_{dataset_type}_autumn_data_{n_memb}memb.csv"
    )

    season_images = [winter_images, spring_images, summer_images, autumn_images]
    season_time = winter_time, spring_time, summer_time, autumn_time

    return season_images, season_time


# #### 3. Apply to Train and Test Datasets

train_season_images, train_season_time = season_split(
    train_images, train_time, "train", n_memb
)


test_season_images, test_season_time = season_split(
    test_images, test_time, "test", n_memb
)


# #### 4. Apply to Projection Datasets

for scenario in scenarios:
    proj_images = np.load(f"../input/preprocessed_2d_proj{scenario}_data_allssp.npy")
    proj_time = pd.read_csv("../input/dates_proj_data.csv")

    proj_season_images, proj_season_time = season_split(
        proj_images, proj_time, "proj", n_memb, scenario
    )
