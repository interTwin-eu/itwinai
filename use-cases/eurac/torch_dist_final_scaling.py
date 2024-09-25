import argparse

import torch
from hython.datasets.datasets import get_dataset
from hython.models.cudnnLSTM import CuDNNLSTM
from hython.normalizer import Normalizer
from hython.sampler import RegularIntervalDownsampler
from hython.utils import read_from_zarr
from itwinai.loggers import  MLFlowLogger
from itwinai.torch.config import TrainingConfiguration
from trainer import RNNDistributedTrainer

# PARAMETERS
EXPERIMENT = "test"
SURROGATE_INPUT = "/p/scratch/intertwin/datasets/eurac/input/adg1km_eobs_preprocessed.zarr/"
SURROGATE_MODEL_OUTPUT = f"/p/scratch/intertwin/datasets/eurac/model/{EXPERIMENT}.pt"
TMP_STATS = "/p/scratch/intertwin/datasets/eurac/stats"

# train/test temporal range
train_temporal_range = slice("2016-01-01", "2018-12-31")
test_temporal_range = slice("2019-01-01", "2020-12-31")

# variables
dynamic_names = ["precip", "pet", "temp"]
static_names = ['thetaS', 'thetaR', 'RootingDepth', 'Swood', 'KsatVer', "Sl"]

DONWSAMPLING = False

# names depends on preprocessing application
mask_names = ["mask_missing", "mask_lake"]

DATASET = "LSTMDataset"

HIDDEN_SIZE = 24
DYNAMIC_INPUT_SIZE = len(dynamic_names)
STATIC_INPUT_SIZE = len(static_names)
TEMPORAL_SUBSET = [150, 150]


def main():
    parser = argparse.ArgumentParser(description='PyTorch LSTM Example')
    parser.add_argument(
            '--batch-size', 
            type=int, 
            default=256,
            help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--strategy', type=str, default='ddp',
                        help='distributed strategy (default=ddp)')
    parser.add_argument('--seed', type=int, default=1696, help='random seed (default: 1696)')
    parser.add_argument(
        '--ckpt-interval', type=int, default=1,
        help='how many batches to wait before logging training status')
    parser.add_argument('--seq_length', type=int, default=60,
                        help='sequence length (default: 60)')
    parser.add_argument('--distributed', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--temporal_subsampling',
                        action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    target_names = ["vwc", "actevap"]
    output_size = len(target_names)

    # Dataset preparation
    Xd = (
        read_from_zarr(url=SURROGATE_INPUT, group="xd", multi_index="gridcell")
        .sel(time=train_temporal_range)
        .xd.sel(feat=dynamic_names)
    )
    Xs = read_from_zarr(
        url=SURROGATE_INPUT, group="xs", multi_index="gridcell").xs.sel(
        feat=static_names
    )
    Y = (
        read_from_zarr(url=SURROGATE_INPUT, group="y", multi_index="gridcell")
        .sel(time=train_temporal_range)
        .y.sel(feat=target_names)
    )

    SHAPE = Xd.attrs["shape"]

    Y_test = (
        read_from_zarr(url=SURROGATE_INPUT, group="y", multi_index="gridcell")
        .sel(time=test_temporal_range)
        .y.sel(feat=target_names)
    )
    Xd_test = (
        read_from_zarr(url=SURROGATE_INPUT, group="xd", multi_index="gridcell")
        .sel(time=test_temporal_range)
        .xd.sel(feat=dynamic_names)
    )

    masks = (
        read_from_zarr(url=SURROGATE_INPUT, group="mask")
        .mask.sel(mask_layer=mask_names)
        .any(dim="mask_layer")
    )

    if DONWSAMPLING:
        train_downsampler = RegularIntervalDownsampler(
            intervals=[3, 3], origin=[0, 0]
        )       
        test_downsampler = RegularIntervalDownsampler(
            intervals=[3, 3], origin=[2, 2]
        )
    else:
        train_downsampler, test_downsampler = None, None

    normalizer_dynamic = Normalizer(method="standardize",
                                    type="spacetime", axis_order="NTC",
                                    save_stats=f"{TMP_STATS}/{EXPERIMENT}_xd.npy")
    normalizer_static = Normalizer(method="standardize",
                                   type="space", axis_order="NTC",
                                   save_stats=f"{TMP_STATS}/{EXPERIMENT}_xs.npy")
    normalizer_target = Normalizer(method="standardize", type="spacetime",
                                   axis_order="NTC",
                                   save_stats=f"{TMP_STATS}/{EXPERIMENT}_y.npy")

    train_dataset = get_dataset(DATASET)(
            Xd,
            Y,
            Xs,
            original_domain_shape=SHAPE,
            mask=masks,
            downsampler=train_downsampler,
            normalizer_dynamic=normalizer_dynamic,
            normalizer_static=normalizer_static,
            normalizer_target=normalizer_target
    )
    validation_dataset = get_dataset(DATASET)(
            Xd_test,
            Y_test,
            Xs,
            original_domain_shape=SHAPE,
            mask=masks,
            downsampler=test_downsampler,
            normalizer_dynamic=normalizer_dynamic,
            normalizer_static=normalizer_static,
            normalizer_target=normalizer_target
    )

    # Model
    model = CuDNNLSTM(
        hidden_size=HIDDEN_SIZE,
        dynamic_input_size=DYNAMIC_INPUT_SIZE,
        static_input_size=STATIC_INPUT_SIZE,
        output_size=output_size
    )

    # Training configuration
    training_config = TrainingConfiguration(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        experiment=EXPERIMENT,
        temporal_subsampling=args.temporal_subsampling,
        temporal_subset=TEMPORAL_SUBSET,
        seq_length=args.seq_length,
        target_names=target_names,
        train_temporal_range=slice("2016-01-01", "2018-12-31"),
        test_temporal_range=slice("2019-01-01", "2020-12-31"),
        dp_weights=SURROGATE_MODEL_OUTPUT,
        distributed=args.distributed,
        num_workers_dataloader=1,
    )

    # Logger
    logger = MLFlowLogger(experiment_name='Distributed Eurac Use case', log_freq=10)

    # Trainer
    trainer = RNNDistributedTrainer(
        config=training_config,
        model=model,
        strategy=args.strategy,
        epochs=args.epochs,
        random_seed=args.seed,
        logger=logger
    )

    # Launch training
    _, _, _, trained_model = trainer.execute(
        train_dataset, validation_dataset, None)


if __name__ == '__main__':
    main()
