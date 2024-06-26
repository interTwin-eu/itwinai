"""
Simplified training script: data generation + training in one
procedural script. This is an INTERMEDIATE step of integration in itwinai.
"""

import torch
import numpy as np
import xarray as xr
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader



from hython.datasets.datasets import get_dataset
from hython.trainer import train_val
from hython.sampler import SamplerBuilder
from hython.metrics import MSEMetric
from hython.losses import RMSELoss
from hython.utils import read_from_zarr, set_seed
from hython.models.cudnnLSTM import CuDNNLSTM
from hython.trainer import RNNTrainer, RNNTrainParams
from hython.normalizer import Normalizer


import matplotlib.pyplot as plt



# PARAMETERS 

EXPERIMENT  = "test"

SURROGATE_INPUT = "https://eurac-eo.s3.amazonaws.com/INTERTWIN/SURROGATE_INPUT/adg1km_eobs_preprocessed.zarr/"

SURROGATE_MODEL_OUTPUT = f"./tmp/{EXPERIMENT}.pt"
TMP_STATS = "./tmp"

# === FILTER ==============================================================

# train/test temporal range
train_temporal_range = slice("2016-01-01","2018-12-31")
test_temporal_range = slice("2019-01-01", "2020-12-31")

# variables
dynamic_names = ["precip", "pet", "temp"] 
static_names = [ 'thetaS', 'thetaR', 'RootingDepth', 'Swood','KsatVer', "Sl"] 
target_names = [ "vwc", "actevap"]

DONWSAMPLING = False

# === MASK ========================================================================================

mask_names = ["mask_missing", "mask_lake"] # names depends on preprocessing application

# === DATASET ========================================================================================

DATASET = "LSTMDataset" # "XBatchDataset"

# == MODEL  ========================================================================================

HIDDEN_SIZE = 24
DYNAMIC_INPUT_SIZE = len(dynamic_names)
STATIC_INPUT_SIZE = len(static_names)
OUTPUT_SIZE = len(target_names)
TARGET_WEIGHTS = {t:1/len(target_names) for t in target_names}



# === SAMPLER/TRAINER ===================================================================================

SEED = 1696
EPOCHS = 20
BATCH = 256
TEMPORAL_SUBSAMPLING = True
TEMPORAL_SUBSET = [150, 150] 
SEQ_LENGTH = 60

assert sum(v for v in TARGET_WEIGHTS.values()) == 1, "check target weights"


if __name__ == "__main__":

    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # === READ TRAIN ===================================================================
    Xd = (
        read_from_zarr(url=SURROGATE_INPUT , group="xd", multi_index="gridcell")
        .sel(time=train_temporal_range)
        .xd.sel(feat=dynamic_names)
    )
    Xs = read_from_zarr(url=SURROGATE_INPUT , group="xs", multi_index="gridcell").xs.sel(
        feat=static_names
    )
    Y = (
        read_from_zarr(url=SURROGATE_INPUT , group="y", multi_index="gridcell")
        .sel(time=train_temporal_range)
        .y.sel(feat=target_names)
    )

    SHAPE = Xd.attrs["shape"]

    # === READ TEST ===================================================================

    Y_test = (
        read_from_zarr(url=SURROGATE_INPUT , group="y", multi_index="gridcell")
        .sel(time=test_temporal_range)
        .y.sel(feat=target_names)
    )
    Xd_test = (
        read_from_zarr(url=SURROGATE_INPUT , group="xd", multi_index="gridcell")
        .sel(time=test_temporal_range)
        .xd.sel(feat=dynamic_names)
    )

    # === READ MASK ===================================================================

    masks = (
        read_from_zarr(url=SURROGATE_INPUT, group="mask")
        .mask.sel(mask_layer=mask_names)
        .any(dim="mask_layer")
    )

    # === SAMPLER ===================================================================

    if DONWSAMPLING:
        train_sampler_builder = SamplerBuilder(sampling_method= "downsampling_regular", 
                                            sampling_method_kwargs = {"intervals": [4,4], "origin": [0, 0]},
                                            minibatch_sampling="random", 
                                            processing="single-gpu")

        test_sampler_builder = SamplerBuilder(sampling_method= "downsampling_regular", 
                                            sampling_method_kwargs = {"intervals": [4,4], "origin": [2, 2]}, 
                                            minibatch_sampling="sequential", 
                                            processing="single-gpu")
    else:
        train_sampler_builder = SamplerBuilder(sampling_method= "default", 
                                            minibatch_sampling="random", 
                                            processing="single-gpu")

        test_sampler_builder = SamplerBuilder(sampling_method= "default", 
                                            minibatch_sampling="sequential", 
                                            processing="single-gpu")
    
    # Initialize samplers, remove indices equivalent to missing values
    train_sampler_builder.initialize(
        shape=SHAPE, mask_missing=masks.values, 
    )
    test_sampler_builder.initialize(
        shape=SHAPE, mask_missing=masks.values, 
    )

    train_sampler = train_sampler_builder.get_sampler()
    test_sampler = test_sampler_builder.get_sampler()


    # === NORMALIZER ===================================================================

    normalizer_dynamic = Normalizer(method="standardize", type="spacetime", shape="1D")

    normalizer_static = Normalizer(method="standardize", type="space", shape="1D")

    normalizer_target = Normalizer(method="standardize", type="spacetime", shape="1D")

    # TODO: precompute and pass statistics as parameters
    normalizer_dynamic.compute_stats(Xd[train_sampler_builder.indices])
    normalizer_static.compute_stats(Xs[train_sampler_builder.indices])
    normalizer_target.compute_stats(Y[train_sampler_builder.indices])

    # save statistics to disk
    Xd = normalizer_dynamic.normalize(Xd, write_to = f"{TMP_STATS}/xd.npy")
    Xs = normalizer_static.normalize(Xs, write_to = f"{TMP_STATS}/xs.npy")
    Y = normalizer_target.normalize(Y, write_to = f"{TMP_STATS}/y.npy")

    # normalize test
    Xd_test = normalizer_dynamic.normalize(Xd_test)
    Y_test = normalizer_target.normalize(Y_test)

    # === DATASET ===================================================================

    train_dataset = get_dataset(DATASET)(
    torch.Tensor(Xd.values), torch.Tensor(Y.values), torch.Tensor(Xs.values)
    )
    test_dataset = get_dataset(DATASET)(
        torch.Tensor(Xd_test.values),
        torch.Tensor(Y_test.values),
        torch.Tensor(Xs.values),
    )

    # === DATA LOADER ===================================================================

    train_loader = DataLoader(train_dataset, batch_size=BATCH , sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH , sampler=test_sampler)


    # === MODEL ===================================================================

    model = CuDNNLSTM(
                    hidden_size=HIDDEN_SIZE, 
                    dynamic_input_size=DYNAMIC_INPUT_SIZE,
                    static_input_size=STATIC_INPUT_SIZE, 
                    output_size=OUTPUT_SIZE
    )

    model.to(device)


    # === TRAIN ===================================================================
    
    opt = optim.Adam(model.parameters(), lr=1e-3)
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


    lepochs = list(range(1, EPOCHS + 1))

    fig, axs = plt.subplots(len(target_names) +1, 1, figsize= (12,10), sharex=True)

    for i, variable in enumerate(target_names):
        axs[i].plot(lepochs, metric_history[f'train_{variable}'], marker='.', linestyle='-', color='b', label='Training')
        axs[i].plot(lepochs, metric_history[f'val_{variable}'], marker='.', linestyle='-', color='r', label='Validation')
        axs[i].set_title(variable)
        axs[i].set_ylabel(metric_fn.__class__.__name__)
        axs[i].grid(True)
        axs[i].legend(bbox_to_anchor=(1,1))

    axs[i+1].plot(lepochs, [i.detach().cpu().numpy() for i in loss_history['train']], marker='.', linestyle='-', color='b', label='Training')
    axs[i+1].plot(lepochs, [i.detach().cpu().numpy() for i in loss_history['val']], marker='.', linestyle='-', color='r', label='Validation')
    axs[i+1].set_title('Loss')
    axs[i+1].set_xlabel('Epochs')
    axs[i+1].set_ylabel(loss_fn.__name__)
    axs[i+1].grid(True)
    axs[i+1].legend(bbox_to_anchor=(1,1))
    plt.show()