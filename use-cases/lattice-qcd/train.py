import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
from normflow import Model
from normflow.nn import DistConvertor_
from normflow.action import ScalarPhi4Action
from normflow.prior import NormalPrior

from itwinai.loggers import ConsoleLogger, MLFlowLogger, LoggersCollection

def make_model():
    net_ = DistConvertor_(10, symmetric=True)
    prior = NormalPrior(shape=(1,))
    action = ScalarPhi4Action(kappa=0, m_sq=-1.2, lambd=0.5)

    mlflow_logger = MLFlowLogger(
        experiment_name="Lattice QCD",
        log_freq="batch"
    )

    mlflow_logger.worker_rank = 0

    model = Model(net_=net_, prior=prior, action=action, logger=mlflow_logger)

    return model

def fit_func(
        model,
        n_epochs=100,
        batch_size=128,
        hyperparam={'fused': True},
    ):
    """Training function to fit model."""
    model.fit(
        n_epochs=n_epochs,
        batch_size=batch_size,
        hyperparam=hyperparam,
    )

def main():
    if torch.cuda.is_available():
        # Initialize distributed backend
        dist.init_process_group(backend="nccl")
        # Get world size to generate unique seeds
        world_size = dist.get_world_size()
        seeds_torch = [torch.randint(2**32 - 1, (1,)).item() for _ in range(world_size)]
        # Create and set up model
        model = make_model()
        # Get rank for seeds
        grank = dist.get_rank()
        # Set seed for current rank
        model.device_handler.set_seed(seeds_torch[grank])
        # Set up model for DDP
        model.device_handler.ddp_wrapper()
        # Train model
        fit_func(model)
        # Destroy distributed process group
        dist.destroy_process_group()
    else:
        # Create and set up model
        model = make_model()
        # Train model
        fit_func(model)

if __name__ == "__main__":
    main()
    sys.exit()
