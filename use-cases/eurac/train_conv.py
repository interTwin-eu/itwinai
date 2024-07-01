"""
Simplified training script: data generation + training in one
procedural script. This is an INTERMEDIATE step of integration in itwinai.
"""

import torch
import numpy as np
import xarray as xr
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DistributedSampler,  DataLoader


# override below
from hython.trainer import train_val

from hython.datasets.datasets import get_dataset
from hython.sampler import SamplerBuilder
from hython.metrics import MSEMetric
from hython.losses import RMSELoss
from hython.utils import read_from_zarr, set_seed
from hython.models.convLSTM import ConvLSTM
from hython.trainer import HythonTrainer, RNNTrainParams
from hython.normalizer import Normalizer


#from itwinai.torch.distributed import (
#    TorchDDPStrategy, NonDistributedStrategy, TorchDistributedStrategy
#)

import matplotlib.pyplot as plt



# PARAMETERS 

EXPERIMENT  = "test"

SURROGATE_INPUT = "https://eurac-eo.s3.amazonaws.com/INTERTWIN/SURROGATE_INPUT/adg1km_eobs_original.zarr/"

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

DATASET = "CubeletsDataset" 
LOAD_IN_MEMORY = True

XSIZE,YSIZE, TSIZE = 60, 60, 60
XOVER,YOVER,TOVER = 10, 10, 10
FILL_MISSING = 0


# == MODEL  ========================================================================================

HIDDEN_SIZE = 24
KERNEL_SIZE = (3, 3)
NUM_LSTM_LAYER = 2
DYNAMIC_INPUT_SIZE = len(dynamic_names)
STATIC_INPUT_SIZE = len(static_names)
OUTPUT_SIZE = len(target_names)
TARGET_WEIGHTS = {t:1/len(target_names) for t in target_names}



# === SAMPLER/TRAINER ===================================================================================

DISTRIBUTED = False

SEED = 1696
EPOCHS = 20
BATCH = 256


assert sum(v for v in TARGET_WEIGHTS.values()) == 1, "check target weights"


# def train_val(
#     trainer,
#     model,
#     train_loader,
#     val_loader,
#     epochs,
#     optimizer,
#     lr_scheduler,
#     dp_weights,
#     strategy: TorchDistributedStrategy
# ):
#     """Override version of hython to support distributed strategy."""

#     import tqdm
#     import copy

#     device = strategy.device

#     loss_history = {"train": [], "val": []}
#     metric_history = {f"train_{target}": []
#                       for target in trainer.P.target_names}
#     metric_history.update({f"val_{target}": []
#                           for target in trainer.P.target_names})

#     best_loss = float("inf")

#     for epoch in tqdm(range(epochs)):

#         # *Added for distributed*
#         train_loader.sampler.set_epoch(epoch)
#         val_loader.sampler.set_epoch(epoch)

#         model.train()

#         # set time indices for training
#         # This has effect only if the trainer overload the
#         # method (i.e. for RNN)
#         trainer.temporal_index([train_loader, val_loader])

#         train_loss, train_metric = trainer.epoch_step(
#             model, train_loader, device, opt=optimizer
#         )

#         # *Added for distributed*
#         # It is a rough way to avoid race conditions among workers
#         # when serializing the best model. More sensible strategies
#         # will come.
#         if strategy.is_main_worker:
#             model.eval()
#             with torch.no_grad():

#                 # set time indices for validation
#                 # This has effect only if the trainer overload the method
#                 # (i.e. for RNN)
#                 trainer.temporal_index([train_loader, val_loader])

#                 val_loss, val_metric = trainer.epoch_step(
#                     model, val_loader, device, opt=None
#                 )

#             lr_scheduler.step(val_loss)

#             loss_history["train"].append(train_loss)
#             loss_history["val"].append(val_loss)

#             for target in trainer.P.target_names:
#                 metric_history[f"train_{target}"].append(train_metric[target])
#                 metric_history[f"val_{target}"].append(val_metric[target])

#             if val_loss < best_loss:
#                 best_loss = val_loss
#                 # The code `best_model_weights` appears to be a variable name in Python. It is not
#                 # assigned any value or operation in the provided snippet, so it is difficult to
#                 # determine its specific purpose without additional context. It could potentially be
#                 # used to store the weights of a machine learning model or any other relevant data
#                 # related to a model.
#                 best_model_weights = copy.deepcopy(model.state_dict())
#                 trainer.save_weights(model, dp_weights)
#                 print("Copied best model weights!")

#             print(f"train loss: {train_loss}")
#             print(f"val loss: {val_loss}")

#             model.load_state_dict(best_model_weights)

#     # NOTE: best model weights are returned only on main worker!
#     return model, loss_history, metric_history



if __name__ == "__main__":

    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # === READ TRAIN ===================================================================
    Xd = read_from_zarr(SURROGATE_INPUT, group="xd").sel(time=train_temporal_range)[dynamic_names]
    Xs = read_from_zarr(SURROGATE_INPUT, group="xs")[static_names]
    Y = read_from_zarr(SURROGATE_INPUT,  group= "y").sel(time=train_temporal_range)[target_names]

    SHAPE = Xd[dynamic_names[0]].shape

    # === READ TEST ===================================================================

    Y_test = (
        read_from_zarr(url=SURROGATE_INPUT , group="y")
        .sel(time=test_temporal_range)[target_names]
    )
    Xd_test = (
        read_from_zarr(url=SURROGATE_INPUT , group="xd")
        .sel(time=test_temporal_range)[dynamic_names]
    )

    # === READ MASK ===================================================================

    masks = (
        read_from_zarr(url=SURROGATE_INPUT, group="mask")
        .mask.sel(mask_layer=mask_names)
        .any(dim="mask_layer")
    )

    if DONWSAMPLING:
        raise NotImplementedError("Downsampling not yet implemented")
    else:
        train_downsampler,test_downsampler = None,None

    # === NORMALIZE ======================================================================

    normalizer_dynamic = Normalizer(method="standardize", type="spacetime", axis_order = "xarray_dataset", save_stats= f"{TMP_STATS}/xd.npy")
    normalizer_static = Normalizer(method="standardize", type="space", axis_order = "xarray_dataset", save_stats= f"{TMP_STATS}/xs.npy")
    normalizer_target = Normalizer(method="standardize", type="spacetime", axis_order = "xarray_dataset", save_stats= f"{TMP_STATS}/y.npy")

    # === DATASET ===================================================================

    train_dataset = get_dataset(DATASET)(
            Xd,
            Y, 
            Xs,
            masks = masks,
            downsampler = train_downsampler,
            normalizer_dynamic = normalizer_dynamic,
            normalizer_static = normalizer_static,
            normalizer_target = normalizer_target,
            shape=SHAPE, # time, lat, lon
            batch_size={"xsize":XSIZE,"ysize":YSIZE,"tsize":TSIZE}, 
            overlap={"xover":XOVER, "yover":YOVER, "tover":TOVER},
            fill_missing=FILL_MISSING,
            persist=LOAD_IN_MEMORY, 
            static_to_dynamic=True
    )
    test_dataset = get_dataset(DATASET)(
            Xd_test,
            Y_test,
            Xs,
            masks = masks,
            downsampler = test_downsampler,
            normalizer_dynamic = normalizer_dynamic,
            normalizer_static = normalizer_static,
            normalizer_target = normalizer_target,
            shape=SHAPE, # time, lat, lon
            batch_size={"xsize":XSIZE,"ysize":YSIZE,"tsize":TSIZE}, 
            overlap={"xover":XOVER, "yover":YOVER, "tover":TOVER},
            fill_missing=FILL_MISSING,
            persist=LOAD_IN_MEMORY, 
            static_to_dynamic=True
    )

    # === SAMPLER ===================================================================

    if DISTRIBUTED:
        # TODO: change with distributed sampler from Iacopo
        train_sampler_builder = DistributedSampler(dataset=train_dataset)
        test_sampler_builder = DistributedSampler(dataset=test_dataset)
    else:
        train_sampler_builder = SamplerBuilder(sampling_method= "cubelets", 
                                            minibatch_sampling="sequential", 
                                            processing="single-gpu")

        test_sampler_builder = SamplerBuilder(sampling_method= "cubelets", 
                                            minibatch_sampling="random", 
                                            processing="single-gpu")

        train_sampler_builder.initialize(
            train_dataset
        ) 
        test_sampler_builder.initialize(
            test_dataset
        )

        train_sampler = train_sampler_builder.get_sampler()
        test_sampler = test_sampler_builder.get_sampler()
    

    # === DATA LOADER ===================================================================

    train_loader = DataLoader(train_dataset, batch_size=BATCH , sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH , sampler=test_sampler)


    # === MODEL ===================================================================

    model = ConvLSTM(
        input_dim =  DYNAMIC_INPUT_SIZE + STATIC_INPUT_SIZE,
        output_dim= OUTPUT_SIZE,
        hidden_dim = (HIDDEN_SIZE),
        kernel_size = KERNEL_SIZE,
        num_layers = NUM_LSTM_LAYER,
        batch_first = True,
        bias = True,
        return_all_layers = False
    ).to(device)


    # === TRAIN ===================================================================
    
    opt = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    loss_fn = RMSELoss(target_weight=TARGET_WEIGHTS)
    metric_fn = MSEMetric(target_names=target_names)

    #if DISTRIBUTED:
    #    strategy = TorchDDPStrategy(backend='nccl')
    #else:
    #    strategy = NonDistributedStrategy()
    
    trainer = HythonTrainer(
        RNNTrainParams(
                experiment=EXPERIMENT,
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