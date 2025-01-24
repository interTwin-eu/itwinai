import torch
import torch.distributed as dist
import time
from torchinfo import summary

from normflow import Model
from normflow.nn import DistConvertor_
from normflow.action import ScalarPhi4Action
from normflow.prior import NormalPrior
from examples.scalar_affine import assemble_net

# from itwinai.loggers import MLFlowLogger

def make_model(lat_shape):
    # net_ = DistConvertor_(10, symmetric=True)
    net_ = assemble_net(lat_shape=lat_shape)
    prior = NormalPrior(shape=lat_shape)
    action = ScalarPhi4Action(kappa=0.67, m_sq=0.67*4, lambd=0.5)

    model = Model(net_=net_, prior=prior, action=action)
    return model

def main():
    hyperparams = {"fused": True}
    n_epochs = 1000
    batch_size = 1024*4
    lat_shape=(8, 8)

    input_shape = (batch_size, *lat_shape)

    if torch.cuda.is_available():
        # Initialize distributed backend
        dist.init_process_group(backend="nccl")

        lwsize = torch.cuda.device_count()
        gwsize = dist.get_world_size()
        grank = dist.get_rank()
        lrank = dist.get_rank() % lwsize

        batch_size = int((batch_size / gwsize))
        input_shape = (batch_size, *lat_shape)

        start_time: float | None = None

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu',lrank)
        torch.cuda.set_device(lrank)

        if dist.get_rank() == 0:
            seeds_torch = [torch.randint(2**32 - 1, (1,)).item() for _ in range(gwsize)]
            print(f"Generated seeds for workers: {seeds_torch}")
        else:
            seeds_torch = [None] * gwsize

        dist.broadcast_object_list(seeds_torch, src=0)
        print(f"Rank {dist.get_rank()} received seeds: {seeds_torch}")

        model = make_model(lat_shape)

        # Log the seed for the current worker
        print(f"Worker {grank} seed: {seeds_torch[grank]}")

        # Set seed for current rank
        model.device_handler.set_seed(seeds_torch[grank])
        # Set up model for DDP
        model.device_handler.ddp_wrapper()

        if grank == 0: 
            start_time = time.time()

        model.fit(
            n_epochs=n_epochs,
            batch_size=batch_size,
            hyperparam=hyperparams,
        )

        if grank == 0: 
            assert start_time is not None
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total training time: {total_time:.2f} seconds.")

        # Destroy distributed process group
        dist.destroy_process_group()
    else:
        model = make_model(prior_size=prior_size)
        summary(model.net_[1], input_shape=input_shape)


if __name__ == "__main__":
    main()
