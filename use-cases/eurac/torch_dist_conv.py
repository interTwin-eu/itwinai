"""
Simplified training script: data generation + training in one
procedural script. This is an INTERMEDIATE step of integration in itwinai.
"""

import matplotlib.pyplot as plt
import torch
from hython.datasets.datasets import get_dataset
from hython.models.convLSTM import ConvLSTM
from hython.normalizer import Normalizer
from hython.sampler import CubeletsDownsampler
from hython.utils import read_from_zarr, set_seed
from trainer import ConvRNNDistributedTrainer

from itwinai.loggers import MLFlowLogger
from itwinai.torch.config import TrainingConfiguration

# PARAMETERS

EXPERIMENT = "convlstm"

RUN_NAME = "convlstm_test"

SURROGATE_INPUT = "/p/scratch/intertwin/datasets/eurac/input/adg1km_eobs_original.zarr"

SURROGATE_MODEL_OUTPUT = f"/p/scratch/intertwin/datasets/eurac/model/{EXPERIMENT}.pt"
TMP_STATS = "/p/scratch/intertwin/datasets/eurac/stats"

# === FILTER ==============================================================

# train/test temporal range
train_temporal_range = slice("2012-01-01", "2018-12-31")
test_temporal_range = slice("2019-01-01", "2020-12-31")

# variables
dynamic_names = ["precip", "pet", "temp"]
static_names = ["thetaS", "thetaR", "f","KsatVer","RootingDepth"]
target_names = ["vwc"]


# === MASK ========================================================================================

mask_names = ["mask_missing", "mask_lake"]  # names depends on preprocessing application

# === DATASET ========================================================================================


DATASET = "CubeletsDataset"

# size of the sample, in height (YSIZE), width (XSIZE) and time (TSIZE)
XSIZE, YSIZE, TSIZE = 10, 10, 60
# sample pixel overlapping
XOVER, YOVER, TOVER = 5, 5, 50

# Criteria for keeping/removing missing or nodata.
# Available options are:
# - max fraction of missing data tolerated in each sample
# - if "any" missing in a sample remove it from the pool
# - if "all" missing in a sample remove it from the pool
MISSING_POLICY = 0.05  # "any", "all"
FILL_MISSING = 0  # nan are converted to whatever value defined here


# == MODEL  ========================================================================================

HIDDEN_SIZE = 36
DYNAMIC_INPUT_SIZE = len(dynamic_names)
STATIC_INPUT_SIZE = len(static_names)
KERNEL_SIZE = (3, 3)
NUM_LSTM_LAYER = 2
OUTPUT_SIZE = len(target_names)

TARGET_WEIGHTS = {t: 1 / len(target_names) for t in target_names}


# === SAMPLER/TRAINER ===================================================================================

# If true the samples are downsampled in either temporal and spatial dimension (or both) by a defined fraction
# The first value in the list correspond to the train samples and the second to the test samples
DONWSAMPLING = True
TEMPORAL_FRAC = [0.2, 0.5]  # train, test
SPATIAL_FRAC = [1, 1]  # train, test


SEED = 42
EPOCHS = 30
BATCH = 16

DISTRIBUTED = True

STRATEGY = "ddp"

assert sum(v for v in TARGET_WEIGHTS.values()) == 1, "check target weights"

if __name__ == "__main__":

    # === READ TRAIN ===================================================================
    Xd = read_from_zarr(SURROGATE_INPUT, group="xd").sel(time=train_temporal_range)[
        dynamic_names
    ]
    Xs = read_from_zarr(SURROGATE_INPUT, group="xs")[static_names]
    Y = read_from_zarr(SURROGATE_INPUT, group="y").sel(time=train_temporal_range)[
        target_names
    ]

    # === READ TEST ===================================================================

    Y_test = read_from_zarr(url=SURROGATE_INPUT, group="y").sel(
        time=test_temporal_range
    )[target_names]
    Xd_test = read_from_zarr(url=SURROGATE_INPUT, group="xd").sel(
        time=test_temporal_range
    )[dynamic_names]

    # === READ MASK ===================================================================

    masks = (
        read_from_zarr(url=SURROGATE_INPUT, group="mask")
        .mask.sel(mask_layer=mask_names)
        .any(dim="mask_layer")
    )

    # === DOWNSAMPLING ===============================================================
    if DONWSAMPLING:
        train_downsampler = CubeletsDownsampler(
            temporal_downsample_fraction=TEMPORAL_FRAC[0],
            spatial_downsample_fraction=SPATIAL_FRAC[0],
        )
        test_downsampler = CubeletsDownsampler(
            temporal_downsample_fraction=TEMPORAL_FRAC[-1],
            spatial_downsample_fraction=SPATIAL_FRAC[-1],
        )
    else:
        train_downsampler, test_downsampler = None, None

    # === NORMALIZE ======================================================================

    normalizer_dynamic = Normalizer(
        method="standardize",
        type="spacetime",
        axis_order="xarray_dataset",
        #save_stats=f"{TMP_STATS}/{EXPERIMENT}_xd.nc",
    )
    normalizer_static = Normalizer(
        method="standardize",
        type="space",
        axis_order="xarray_dataset",
        #save_stats=f"{TMP_STATS}/{EXPERIMENT}_xs.nc",
    )
    normalizer_target = Normalizer(
        method="standardize",
        type="spacetime",
        axis_order="xarray_dataset",
        #save_stats=f"{TMP_STATS}/{EXPERIMENT}_y.nc",
    )

    # === DATASET ===================================================================

    train_dataset = get_dataset(DATASET)(
        Xd,
        Y,
        Xs,
        mask=masks,
        downsampler=train_downsampler,
        normalizer_dynamic=normalizer_dynamic,
        normalizer_static=normalizer_static,
        normalizer_target=normalizer_target,
        shape=Xd.precip.shape,  # time, lat, lon
        batch_size={"xsize": XSIZE, "ysize": YSIZE, "tsize": TSIZE},
        overlap={"xover": XOVER, "yover": YOVER, "tover": TOVER},
        missing_policy=MISSING_POLICY,
        fill_missing=FILL_MISSING,
        persist=True,
        static_to_dynamic=True
    )
    validation_dataset = get_dataset(DATASET)(
        Xd_test,
        Y_test,
        Xs,
        mask=masks,
        downsampler=test_downsampler,
        normalizer_dynamic=normalizer_dynamic,
        normalizer_static=normalizer_static,
        normalizer_target=normalizer_target,
        shape=Xd_test.precip.shape,  # time, lat, lon
        batch_size={"xsize": XSIZE, "ysize": YSIZE, "tsize": TSIZE},
        overlap={"xover": XOVER, "yover": YOVER, "tover": TOVER},
        missing_policy=MISSING_POLICY,
        fill_missing=FILL_MISSING,
        persist=True,
        static_to_dynamic=True
    )


    # === MODEL ===================================================================

    model = ConvLSTM(
        input_dim=DYNAMIC_INPUT_SIZE + STATIC_INPUT_SIZE,
        output_dim=OUTPUT_SIZE,
        hidden_dim=(HIDDEN_SIZE),
        kernel_size=KERNEL_SIZE,
        num_layers=NUM_LSTM_LAYER,
        batch_first=True,
        bias=False,
        return_all_layers=False,
    )

    # === TRAIN ===================================================================

    # Training configuration
    training_config = TrainingConfiguration(
        batch_size=BATCH,
        lr=0.001,
        epochs=EPOCHS,
        experiment=EXPERIMENT,
        rnn_config = dict(
            temporal_subsampling=False,
            temporal_subset=1,
            target_names=target_names,
            train_temporal_range=train_temporal_range,
            test_temporal_range=test_temporal_range,
            dp_weights=SURROGATE_MODEL_OUTPUT,
            distributed=DISTRIBUTED
        )
    )


    logger = MLFlowLogger(experiment_name=f"EXP_{EXPERIMENT}",
                          run_name=RUN_NAME,
                          log_freq=10)

    trainer = ConvRNNDistributedTrainer(
        config=training_config,
        model=model,
        strategy=STRATEGY,
        epochs=EPOCHS,
        random_seed=SEED,
        logger=logger
    )

    # Launch training
    _, _, _, trained_model = trainer.execute(
        train_dataset, validation_dataset, None)


