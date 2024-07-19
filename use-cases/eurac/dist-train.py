"""
Simplified training script: data generation + training in one
procedural script. This is an INTERMEDIATE step of integration in itwinai.
"""

import torch
# import numpy as np
# import xarray as xr
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DistributedSampler


from hython.datasets.datasets import get_dataset

# override below
# from hython.trainer import train_val

from hython.sampler import SamplerBuilder, RegularIntervalDownsampler
from hython.metrics import MSEMetric
from hython.losses import RMSELoss
from hython.utils import read_from_zarr, set_seed
from hython.models.cudnnLSTM import CuDNNLSTM
from hython.trainer import RNNTrainer, RNNTrainParams
from hython.normalizer import Normalizer

from itwinai.torch.distributed import (
    TorchDDPStrategy, NonDistributedStrategy, TorchDistributedStrategy
)

import matplotlib.pyplot as plt


# PARAMETERS

EXPERIMENT = "test"

SURROGATE_INPUT = "/p/scratch/intertwin/datasets/eurac/input/adg1km_eobs_preprocessed.zarr/"

SURROGATE_MODEL_OUTPUT = f"/p/scratch/intertwin/datasets/eurac/model/{EXPERIMENT}.pt"
TMP_STATS = "/p/scratch/intertwin/datasets/eurac/stats"

# === FILTER ==============================================================

# train/test temporal range
train_temporal_range = slice("2016-01-01", "2018-12-31")
test_temporal_range = slice("2019-01-01", "2020-12-31")

# variables
dynamic_names = ["precip", "pet", "temp"]
static_names = ['thetaS', 'thetaR', 'RootingDepth', 'Swood', 'KsatVer', "Sl"]
target_names = ["vwc", "actevap"]

DONWSAMPLING = False

# === MASK ==================================================================

# names depends on preprocessing application
mask_names = ["mask_missing", "mask_lake"]

# === DATASET ===============================================================

DATASET = "LSTMDataset" 

# == MODEL  ==================================================================

HIDDEN_SIZE = 24
DYNAMIC_INPUT_SIZE = len(dynamic_names)
STATIC_INPUT_SIZE = len(static_names)
OUTPUT_SIZE = len(target_names)
TARGET_WEIGHTS = {t: 1/len(target_names) for t in target_names}


# === SAMPLER/TRAINER ========================================================

DISTRIBUTED = True

SEED = 1696
EPOCHS = 20
BATCH = 256
TEMPORAL_SUBSAMPLING = True
TEMPORAL_SUBSET = [150, 150]
SEQ_LENGTH = 60


assert sum(v for v in TARGET_WEIGHTS.values()) == 1, "check target weights"


def train_val(
    trainer,
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    lr_scheduler,
    dp_weights,
    strategy: TorchDistributedStrategy
):
    """Override version of hython to support distributed strategy."""

    import tqdm
    import copy

    device = strategy.device

    loss_history = {"train": [], "val": []}
    metric_history = {f"train_{target}": []
                      for target in trainer.P.target_names}
    metric_history.update({f"val_{target}": []
                          for target in trainer.P.target_names})

    best_loss = float("inf")

    for epoch in tqdm(range(epochs)):

        # *Added for distributed*
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        model.train()

        # set time indices for training
        # This has effect only if the trainer overload the
        # method (i.e. for RNN)
        trainer.temporal_index([train_loader, val_loader])

        train_loss, train_metric = trainer.epoch_step(
            model, train_loader, device, opt=optimizer
        )

        # *Added for distributed*
        # It is a rough way to avoid race conditions among workers
        # when serializing the best model. More sensible strategies
        # will come.
        if strategy.is_main_worker:
            model.eval()
            with torch.no_grad():

                # set time indices for validation
                # This has effect only if the trainer overload the method
                # (i.e. for RNN)
                trainer.temporal_index([train_loader, val_loader])

                val_loss, val_metric = trainer.epoch_step(
                    model, val_loader, device, opt=None
                )

            lr_scheduler.step(val_loss)

            loss_history["train"].append(train_loss)
            loss_history["val"].append(val_loss)

            for target in trainer.P.target_names:
                metric_history[f"train_{target}"].append(train_metric[target])
                metric_history[f"val_{target}"].append(val_metric[target])

            if val_loss < best_loss:
                best_loss = val_loss
                # The code `best_model_weights` appears to be a variable name in Python. It is not
                # assigned any value or operation in the provided snippet, so it is difficult to
                # determine its specific purpose without additional context. It could potentially be
                # used to store the weights of a machine learning model or any other relevant data
                # related to a model.
                best_model_weights = copy.deepcopy(model.state_dict())
                trainer.save_weights(model, dp_weights)
                print("Copied best model weights!")

            print(f"train loss: {train_loss}")
            print(f"val loss: {val_loss}")

            model.load_state_dict(best_model_weights)

    # NOTE: best model weights are returned only on main worker!
    return model, loss_history, metric_history


if __name__ == "__main__":

    if DISTRIBUTED:
        strategy = TorchDDPStrategy(backend='nccl')
    else:
        strategy = NonDistributedStrategy()

    strategy.init()

    set_seed(SEED)

    # moved below
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # === READ TRAIN =========================================================
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

    # === READ TEST ==========================================================

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

    # === READ MASK =========================================================

    masks = (
        read_from_zarr(url=SURROGATE_INPUT, group="mask")
        .mask.sel(mask_layer=mask_names)
        .any(dim="mask_layer")
    )

    # === DOWNSAMPLING =================================================================

    if DONWSAMPLING:
       train_downsampler = RegularIntervalDownsampler(
            intervals=[3,3], origin=[0,0]
        )       
       test_downsampler = RegularIntervalDownsampler(
            intervals=[3,3], origin=[2,2]
        )
    else:
        train_downsampler,test_downsampler = None,None

    # === NORMALIZE ======================================================================

    normalizer_dynamic = Normalizer(method="standardize", type="spacetime", axis_order = "NTC", save_stats= f"{TMP_STATS}/{EXPERIMENT}_xd.npy")
    normalizer_static = Normalizer(method="standardize", type="space", axis_order = "NTC", save_stats= f"{TMP_STATS}/{EXPERIMENT}_xs.npy")
    normalizer_target = Normalizer(method="standardize", type="spacetime", axis_order = "NTC", save_stats= f"{TMP_STATS}/{EXPERIMENT}_y.npy")



    # === DATSET =======================================================================

    train_dataset = get_dataset(DATASET)(
            Xd,
            Y,
            Xs,
            original_domain_shape=SHAPE,
            mask = masks,
            downsampler = train_downsampler,
            normalizer_dynamic = normalizer_dynamic,
            normalizer_static = normalizer_static,
            normalizer_target = normalizer_target
    )
    test_dataset = get_dataset(DATASET)(
            Xd_test,
            Y_test,
            Xs,
            original_domain_shape=SHAPE,
            mask = masks,
            downsampler = test_downsampler,
            normalizer_dynamic = normalizer_dynamic,
            normalizer_static = normalizer_static,
            normalizer_target = normalizer_target
    )

    # === SAMPLER ============================================================

    train_sampler_builder = SamplerBuilder(
        train_dataset,
        sampling="random", 
        processing="multi-gpu" if DISTRIBUTED else "single-gpu")

    test_sampler_builder = SamplerBuilder(
        test_dataset,
        sampling="sequential", 
        processing="multi-gpu" if DISTRIBUTED else "single-gpu")


    train_sampler = train_sampler_builder.get_sampler()
    test_sampler = test_sampler_builder.get_sampler()



    # === MODEL OPTIM LOSS ================================================

    model = CuDNNLSTM(
        hidden_size=HIDDEN_SIZE,
        dynamic_input_size=DYNAMIC_INPUT_SIZE,
        static_input_size=STATIC_INPUT_SIZE,
        output_size=OUTPUT_SIZE
    )

    # model.to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    loss_fn = RMSELoss(target_weight=TARGET_WEIGHTS)
    metric_fn = MSEMetric(target_names=target_names)

    # Distributed model
    device = strategy.device

    model, opt, lr_scheduler = strategy.distributed(
        model=model, optimizer=opt, lr_scheduler=lr_scheduler
    )

    # Distributed dataloader

    train_loader = strategy.create_dataloader(
        train_dataset, batch_size=BATCH, sampler=train_sampler)
    test_loader = strategy.create_dataloader(
        test_dataset, batch_size=BATCH, sampler=test_sampler)

    # === TRAIN ==============================================================

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

    model, loss_history, metric_history = train_val(
        trainer,
        model,
        train_loader,
        test_loader,
        EPOCHS,
        opt,
        lr_scheduler,
        SURROGATE_MODEL_OUTPUT,
        strategy=strategy
    )

    if strategy.is_main_worker:

        lepochs = list(range(1, EPOCHS + 1))

        fig, axs = plt.subplots(len(target_names) +1, 1, figsize= (12,10), sharex=True)

        axs[0].plot(lepochs, [i.detach().cpu().numpy() for i in loss_history['train']], marker='.', linestyle='-', color='b', label='Training')
        axs[0].plot(lepochs, [i.detach().cpu().numpy() for i in loss_history['val']], marker='.', linestyle='-', color='r', label='Validation')
        axs[0].set_title('Loss')
        axs[0].set_ylabel(loss_fn.__name__)
        axs[0].grid(True)
        axs[0].legend(bbox_to_anchor=(1,1))

        for i, variable in enumerate(target_names):
            axs[i+1].plot(lepochs, metric_history[f'train_{variable}'], marker='.', linestyle='-', color='b', label='Training')
            axs[i+1].plot(lepochs, metric_history[f'val_{variable}'], marker='.', linestyle='-', color='r', label='Validation')
            axs[i+1].set_title(variable)
            axs[i+1].set_ylabel(metric_fn.__class__.__name__)
            axs[i+1].grid(True)
            axs[i+1].legend(bbox_to_anchor=(1,1))

    # End distributed
    strategy.clean_up()
