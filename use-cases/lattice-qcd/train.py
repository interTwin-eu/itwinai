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

        lwsize = torch.cuda.device_count()
        gwsize = dist.get_world_size()
        grank = dist.get_rank()
        lrank = dist.get_rank()%lwsize

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu',lrank)
        torch.cuda.set_device(lrank)

        if dist.get_rank() == 0:
            seeds_torch = [torch.randint(2**32 - 1, (1,)).item() for _ in range(gwsize)]
            print(f"Generated seeds for workers: {seeds_torch}")
        else:
            seeds_torch = [None] * gwsize

        dist.broadcast_object_list(seeds_torch, src=0)
        print(f"Rank {dist.get_rank()} received seeds: {seeds_torch}")

        # Create and set up model
        model = make_model()

        # Log the seed for the current worker
        print(f"Worker {grank} seed: {seeds_torch[grank]}")

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
