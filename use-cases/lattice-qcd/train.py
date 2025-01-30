import torch
import torch.distributed as dist
import time
from torchinfo import summary
from pyinstrument import Profiler

from normflow import Model
from normflow.nn import DistConvertor_
from normflow.action import ScalarPhi4Action
from normflow.prior import NormalPrior
from examples.scalar_affine import assemble_net

# from itwinai.loggers import MLFlowLogger

def make_model(lat_shape):
    net_ = assemble_net(lat_shape=lat_shape)
    prior = NormalPrior(shape=lat_shape)
    action = ScalarPhi4Action(kappa=0.67, m_sq=-0.67*4, lambd=0.5)

    model = Model(net_=net_, prior=prior, action=action)
    return model

def main():
    hyperparams = {"fused": True}
    n_epochs = 100
    batch_size = 1024
    lat_shape=(2, 2)

    model = make_model(lat_shape)
    summary(model.net_, input_shape=lat_shape)

    local_rank = model.device_handler.local_rank
    global_rank = model.device_handler.rank
    global_world_size = model.device_handler.nranks

    if torch.cuda.is_available():
        # Initialize distributed backend
        torch.cuda.set_device(local_rank)

        if global_rank == 0:
            seeds_torch = [torch.randint(2**32 - 1, (1,)).item() for _ in range(global_world_size)]
            print(f"Generated seeds for workers: {seeds_torch}")
        else:
            seeds_torch = [None] * global_world_size

        dist.broadcast_object_list(seeds_torch, src=0)
        print(f"Rank {dist.get_rank()} received seeds: {seeds_torch}")
        print(f"Worker {global_rank} seed: {seeds_torch[global_rank]}")

        model.device_handler.set_seed(seeds_torch[global_rank])
        model.device_handler.ddp_wrapper()

    batch_size = int((batch_size / global_world_size))
    start_time: float | None = None
    if global_rank == 0: 
        start_time = time.time()

    profiler = Profiler()
    profiler.start()
    model.fit(
        n_epochs=n_epochs,
        batch_size=batch_size,
        hyperparam=hyperparams,
    )
    profiler.stop()
    profiler.print(show_all=True)

    if global_rank == 0: 
        assert start_time is not None
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time: {total_time:.2f} seconds.")

    if not torch.cuda.is_available():
        return

    # Destroy distributed process group
    dist.destroy_process_group()



if __name__ == "__main__":
    main()
