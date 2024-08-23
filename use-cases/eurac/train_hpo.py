import torch
import numpy as np
import xarray as xr
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from ray import tune
from ray.air import session
import os

from hython.datasets.datasets import get_dataset
from hython.trainer import train_val
from hython.sampler import SamplerBuilder, RegularIntervalDownsampler
from hython.metrics import MSEMetric
from hython.losses import RMSELoss
from hython.utils import read_from_zarr, set_seed
from hython.models.cudnnLSTM import CuDNNLSTM
from hython.trainer import RNNTrainer, RNNTrainParams
from hython.normalizer import Normalizer

# PARAMETERS
EXPERIMENT = "test"
SURROGATE_INPUT = "/p/scratch/intertwin/datasets/eurac/input/adg1km_eobs_preprocessed.zarr/"
SURROGATE_MODEL_OUTPUT = f"/p/scratch/intertwin/datasets/eurac/model/{EXPERIMENT}.pt"
TMP_STATS = "/p/scratch/intertwin/datasets/eurac/stats"
train_temporal_range = slice("2016-01-01", "2018-12-31")
test_temporal_range = slice("2019-01-01", "2020-12-31")
dynamic_names = ["precip", "pet", "temp"]
static_names = ['thetaS', 'thetaR', 'RootingDepth', 'Swood', 'KsatVer', "Sl"]
target_names = ["vwc", "actevap"]
mask_names = ["mask_missing", "mask_lake"]
DATASET = "LSTMDataset"
HIDDEN_SIZE = 24
DYNAMIC_INPUT_SIZE = len(dynamic_names)
STATIC_INPUT_SIZE = len(static_names)
OUTPUT_SIZE = len(target_names)
TARGET_WEIGHTS = {t: 1/len(target_names) for t in target_names}
SEED = 1696
EPOCHS = 2
TEMPORAL_SUBSAMPLING = True
TEMPORAL_SUBSET = [150, 150]
SEQ_LENGTH = 60
DONWSAMPLING = False


def main(config):
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data processing
    Xd, Xs, Y, Y_test, Xd_test, masks = prepare_data()
    train_dataset, test_dataset = create_datasets(
        Xd, Xs, Y, Xd_test, Y_test, masks)
    train_loader, test_loader = create_data_loaders(
        train_dataset, test_dataset, config['batch_size'])

    # Model setup
    model = CuDNNLSTM(
        hidden_size=HIDDEN_SIZE,
        dynamic_input_size=DYNAMIC_INPUT_SIZE,
        static_input_size=STATIC_INPUT_SIZE,
        output_size=OUTPUT_SIZE
    ).to(device)

    # Training setup
    opt = optim.Adam(model.parameters(), lr=config['lr'])
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    loss_fn = RMSELoss(target_weight=TARGET_WEIGHTS)
    metric_fn = MSEMetric(target_names=target_names)
    trainer = RNNTrainer(
        RNNTrainParams(
            experiment=EXPERIMENT,
            temporal_subsampling=TEMPORAL_SUBSAMPLING,
            temporal_subset=TEMPORAL_SUBSET,
            seq_length=SEQ_LENGTH,
            target_names=target_names,
            metric_func=metric_fn,
            loss_func=loss_fn)
    )

    # Execute training
    model, loss_history, metric_history = train_val(
        trainer,
        model,
        train_loader,
        test_loader,
        EPOCHS,
        opt,
        lr_scheduler,
        SURROGATE_MODEL_OUTPUT,
        device
    )

    # Reporting the mean validation loss to Ray Tune
    # tune.report(loss=np.mean([l.item() for l in loss_history['val']]))
    # air.session.report({"loss": np.mean([l.item() for l in loss_history['val']])})
    session.report({"loss": np.mean([l.item() for l in loss_history['val']])})


def prepare_data():
    # Reading and preparing data as per the original script
    Xd = (read_from_zarr(url=SURROGATE_INPUT, group="xd", multi_index="gridcell").sel(
        time=train_temporal_range).xd.sel(feat=dynamic_names))
    Xs = read_from_zarr(url=SURROGATE_INPUT, group="xs",
                        multi_index="gridcell").xs.sel(feat=static_names)
    Y = (read_from_zarr(url=SURROGATE_INPUT, group="y", multi_index="gridcell").sel(
        time=train_temporal_range).y.sel(feat=target_names))
    Y_test = read_from_zarr(url=SURROGATE_INPUT, group="y", multi_index="gridcell").sel(
        time=test_temporal_range).y.sel(feat=target_names)
    Xd_test = read_from_zarr(url=SURROGATE_INPUT, group="xd", multi_index="gridcell").sel(
        time=test_temporal_range).xd.sel(feat=dynamic_names)
    masks = (read_from_zarr(url=SURROGATE_INPUT, group="mask").mask.sel(
        mask_layer=mask_names).any(dim="mask_layer"))
    return Xd, Xs, Y, Y_test, Xd_test, masks


def create_datasets(Xd, Xs, Y, Xd_test, Y_test, masks):
    # Creating datasets as per the original script
    train_downsampler = None if not DONWSAMPLING else RegularIntervalDownsampler(
        intervals=[3, 3], origin=[0, 0])
    test_downsampler = None if not DONWSAMPLING else RegularIntervalDownsampler(
        intervals=[3, 3], origin=[2, 2])
    normalizer_dynamic = Normalizer(
        method="standardize", type="spacetime", axis_order="NTC", save_stats=f"{TMP_STATS}/xd.npy")
    normalizer_static = Normalizer(
        method="standardize", type="space", axis_order="NTC", save_stats=f"{TMP_STATS}/xs.npy")
    normalizer_target = Normalizer(
        method="standardize", type="spacetime", axis_order="NTC", save_stats=f"{TMP_STATS}/y.npy")

    train_dataset = get_dataset(DATASET)(
        Xd,
        Y,
        Xs,
        original_domain_shape=Xd.attrs["shape"],
        mask=masks,
        downsampler=train_downsampler,
        normalizer_dynamic=normalizer_dynamic,
        normalizer_static=normalizer_static,
        normalizer_target=normalizer_target)
    test_dataset = get_dataset(DATASET)(
        Xd_test,
        Y_test,
        Xs,
        original_domain_shape=Xd.attrs["shape"],
        mask=masks, downsampler=test_downsampler,
        normalizer_dynamic=normalizer_dynamic,
        normalizer_static=normalizer_static,
        normalizer_target=normalizer_target)
    return train_dataset, test_dataset


def create_data_loaders(train_dataset, test_dataset, batch_size):
    train_sampler = SamplerBuilder(
        train_dataset, sampling="random", processing="single-gpu").get_sampler()
    test_sampler = SamplerBuilder(
        test_dataset, sampling="sequential", processing="single-gpu").get_sampler()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, test_loader


# HPO configuration and execution
if __name__ == "__main__":
    config = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([64, 128, 256, 512])
    }
    analysis = tune.run(
        main,
        config=config,
        num_samples=10,
        resources_per_trial={"cpu": 8, "gpu": 1},
        progress_reporter=tune.CLIReporter(
            parameter_columns=["lr", "batch_size"],
            metric_columns=["loss"],
            max_report_frequency=60
        ),
        metric="loss",
        mode="min"
    )

    # Access the results dataframe
    df = analysis.dataframe()
    
    sorted_df = df.sort_values("loss", ascending=True)
    print("Results dataframe sorted by loss:")
    print(sorted_df)
    
    # Optionally, you can also print the best trial's config and loss
    print("Best hyperparameters found were:", analysis.best_config)
    best_trial_loss = sorted_df.iloc[0]['loss']
    print("Best trial loss:", best_trial_loss)