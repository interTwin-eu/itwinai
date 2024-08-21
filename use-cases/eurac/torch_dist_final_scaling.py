import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import torch.nn as nn
import pandas as pd
from timeit import default_timer as timer
import os

from hython.datasets.datasets import get_dataset
from hython.sampler import SamplerBuilder, RegularIntervalDownsampler
from hython.metrics import MSEMetric
from hython.losses import RMSELoss
from hython.utils import read_from_zarr
from hython.models.cudnnLSTM import CuDNNLSTM
from hython.trainer import RNNTrainer, RNNTrainParams
from hython.normalizer import Normalizer
from itwinai.torch.distributed import (
    DeepSpeedStrategy
)

from itwinai.torch.trainer import TorchTrainer
from itwinai.loggers import Logger
from itwinai.torch.type import Metric
from itwinai.loggers import MLFlowLogger, EpochTimeTracker
from typing import (
    Optional, Dict, Union, Literal
)
from itwinai.torch.config import TrainingConfiguration
from tqdm.auto import tqdm
import copy
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
target_names = ["vwc", "actevap"]

DONWSAMPLING = False

# names depends on preprocessing application
mask_names = ["mask_missing", "mask_lake"]

DATASET = "LSTMDataset"

HIDDEN_SIZE = 24
DYNAMIC_INPUT_SIZE = len(dynamic_names)
STATIC_INPUT_SIZE = len(static_names)
OUTPUT_SIZE = len(target_names)
TARGET_WEIGHTS = {t: 1/len(target_names) for t in target_names}
TEMPORAL_SUBSET = [150, 150]

assert sum(v for v in TARGET_WEIGHTS.values()) == 1, "check target weights"


class RNNDistributedTrainer(TorchTrainer):
    """Trainer class for RNN model using pytorch.

    Args:
        config (Union[Dict, TrainingConfiguration]): training configuration
            containing hyperparameters.
        epochs (int): number of training epochs.
        model (Optional[nn.Module], optional): model to train.
            Defaults to None.
        strategy (Literal['ddp', 'deepspeed', 'horovod'], optional):
            distributed strategy. Defaults to 'ddp'.
        validation_every (Optional[int], optional): run a validation epoch
            every ``validation_every`` epochs. Disabled if None. Defaults to 1.
        test_every (Optional[int], optional): run a test epoch
            every ``test_every`` epochs. Disabled if None. Defaults to None.
        random_seed (Optional[int], optional): set random seed for
            reproducibility. If None, the seed is not set. Defaults to None.
        logger (Optional[Logger], optional): logger for ML tracking.
            Defaults to None.
        metrics (Optional[Dict[str, Metric]], optional): map of torch metrics
            metrics. Defaults to None.
        checkpoints_location (str): path to checkpoints directory.
            Defaults to "checkpoints".
        checkpoint_every (Optional[int]): save a checkpoint every
            ``checkpoint_every`` epochs. Disabled if None. Defaults to None.
        name (Optional[str], optional): trainer custom name. Defaults to None.
    """

    def __init__(
            self,
            config: Union[Dict, TrainingConfiguration],
            epochs: int,
            model: Optional[nn.Module] = None,
            strategy: Literal["ddp", "deepspeed"] = 'ddp',
            validation_every: Optional[int] = 1,
            test_every: Optional[int] = None,
            random_seed: Optional[int] = None,
            logger: Optional[Logger] = None,
            metrics: Optional[Dict[str, Metric]] = None,
            checkpoints_location: str = "checkpoints",
            checkpoint_every: Optional[int] = None,
            name: Optional[str] = None, **kwargs) -> None:
        super().__init__(
            config=config,
            epochs=epochs,
            model=model,
            strategy=strategy,
            validation_every=validation_every,
            test_every=test_every,
            random_seed=random_seed,
            logger=logger,
            metrics=metrics,
            checkpoints_location=checkpoints_location,
            checkpoint_every=checkpoint_every,
            name=name,
            **kwargs)
        self.save_parameters(**self.locals2params(locals()))

    def create_model_loss_optimizer(self) -> None:
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.lr
        )
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10)
        self.loss_fn = RMSELoss(target_weight=TARGET_WEIGHTS)
        self.metric_fn = MSEMetric(target_names=target_names)

        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(
                    train_micro_batch_size_per_gpu=self.config.batch_size
                )
            )
        else:
            distribute_kwargs = {}
        # Distribute discriminator and its optimizer
        self.model, self.optimizer, _ = self.strategy.distributed(
            model=self.model, optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler, **distribute_kwargs)

    def train(self):
        st = timer()
        # uncomment all lines relative to accuracy if you want to measure
        # IOU between generated and real spectrograms.
        # Note that it significantly slows down the whole process
        # it also might not work as the function has not been fully
        # implemented yet
        if self.strategy.is_main_worker:
            print('TIMER: broadcast:', timer()-st, 's')
            print('\nDEBUG: start training')
            print('--------------------------------------------------------')
            nnod = os.environ.get('SLURM_NNODES', 'unk')
            s_name = f"{os.environ.get('DIST_MODE', 'unk')}-torch"
            epoch_time_tracker = EpochTimeTracker(
                series_name=s_name,
                csv_file=f"epochtime_{s_name}_{nnod}N.csv"
            )
        """Override version of hython to support distributed strategy."""
        trainer = RNNTrainer(
                RNNTrainParams(
                    experiment=self.config.experiment,
                    temporal_subsampling=self.config.temporal_subsampling,
                    temporal_subset=self.config.temporal_subset,
                    seq_length=self.config.seq_length,
                    target_names=target_names,
                    metric_func=self.metric_fn,
                    loss_func=self.loss_fn)
        )

        device = self.strategy.device()
        loss_history = {"train": [], "val": []}
        metric_history = {f"train_{target}": []
                          for target in trainer.P.target_names}
        metric_history.update({f"val_{target}": []
                              for target in trainer.P.target_names})

        best_loss = float("inf")
        et = timer()
        for epoch in tqdm(range(self.epochs)):
            lt = timer()
            if self.strategy.is_distributed:
                # *Added for distributed*
                self.train_loader.sampler.set_epoch(epoch)
                self.val_loader.sampler.set_epoch(epoch)

            self.model.train()

            # set time indices for training
            # This has effect only if the trainer overload the
            # method (i.e. for RNN)
            trainer.temporal_index([self.train_loader, self.val_loader])

            train_loss, train_metric = trainer.epoch_step(
                self.model, self.train_loader, device, opt=self.optimizer
            )

            self.model.eval()
            with torch.no_grad():
                # set time indices for validation
                # This has effect only if the trainer overload the method
                # (i.e. for RNN)
                trainer.temporal_index([self.train_loader, self.val_loader])

                val_loss, val_metric = trainer.epoch_step(
                    self.model, self.val_loader, device, opt=None
                )

            # gather losses from each worker and place them on the main worker.
            worker_val_losses = self.strategy.gather(val_loss, dst_rank=0)
            if self.strategy.global_rank() == 0:
                avg_val_loss = torch.mean(torch.stack(
                    worker_val_losses)).detach().cpu()
                self.lr_scheduler.step(avg_val_loss)
                loss_history["train"].append(train_loss)
                loss_history["val"].append(avg_val_loss)
                self.log(
                    item=train_loss.item(),
                    identifier='train_loss_per_epoch',
                    kind='metric',
                    step=epoch,
                )
                self.log(
                    item=avg_val_loss.item(),
                    identifier='val_loss_per_epoch',
                    kind='metric',
                    step=epoch,
                )

                for target in trainer.P.target_names:
                    metric_history[f"train_{target}"].append(
                        train_metric[target])
                    metric_history[f"val_{target}"].append(
                        val_metric[target])
                # Aggregate and log metrics
                avg_metrics = pd.DataFrame(metric_history).mean().to_dict()
                for m_name, m_val in avg_metrics.items():
                    self.log(
                        item=m_val,
                        identifier=m_name + '_epoch',
                        kind='metric',
                        step=epoch,
                    )

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    # The code `best_model_weights` appears to be a variable
                    # name in Python. It is not assigned any value
                    # or operation in the provided snippet, so it is
                    # difficult to determine its specific purpose
                    # without additional context. It could potentially be
                    # used to store the weights of a machine learning model
                    # or any other relevant data  related to a model.
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                    trainer.save_weights(self.model, self.config.dp_weights)
                    print("Copied best model weights!")

                    print(f"train loss: {train_loss}")
                    print(f"val loss: {avg_val_loss}")

                self.model.load_state_dict(best_model_weights)
            if self.strategy.is_main_worker:
                print('TIMER: epoch time:', timer()-lt, 's')
                epoch_time_tracker.add_epoch_time(epoch-1, timer()-lt)
        return loss_history, metric_history

    def create_dataloaders(
            self, train_dataset, validation_dataset, test_dataset):
        train_sampler_builder = SamplerBuilder(
            train_dataset,
            sampling="random",
            processing="multi-gpu" if self.config.distributed else "single-gpu")

        val_sampler_builder = SamplerBuilder(
            validation_dataset,
            sampling="sequential",
            processing="multi-gpu" if self.config.distributed else "single-gpu")

        train_sampler = train_sampler_builder.get_sampler()
        val_sampler = val_sampler_builder.get_sampler()

        self.train_loader = self.strategy.create_dataloader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=self.config.pin_gpu_memory,
            generator=self.torch_rng,
            sampler=train_sampler
        )

        if validation_dataset is not None:
            self.val_loader = self.strategy.create_dataloader(
                dataset=validation_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                sampler=val_sampler
            )


def main():
    parser = argparse.ArgumentParser(description='PyTorch LSTM Example')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--strategy', type=str, default='ddp',
                        help='distributed strategy (default=ddp)')
    parser.add_argument('--seed', type=int, default=1696,
                        help='random seed (default: 1696)')
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
        output_size=OUTPUT_SIZE
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
        distributed=args.distributed
    )

    # Logger
    logger = MLFlowLogger(experiment_name='Distributed Eurac Use case',
                          log_freq=10)

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
