import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .reproducibility import seed_worker


def _distributed_dataloader(dataloader: DataLoader, gwsize, grank):
    """Makes a Dataloader distributed."""
    sampler = DistributedSampler(
        dataloader.dataset, num_replicas=gwsize, rank=grank, shuffle=True
    )
    # Recreate dataloader, with updated sampler
    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=seed_worker,  # dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers,
        pin_memory_device=dataloader.pin_memory_device,
    )


def distributed(func):
    """The decorated function must have a standard signature.

    Its first arguments must be:
    model, train_dataloader, validation_dataloader, device (in this order).

    Additional args or kwargs are allowed consistently with the signature
    of the decorated function.
    """

    def dist_train(
        model, train_dataloader, validation_dataloader=None, device="cpu", *args, **kwargs
    ):
        if torch.cuda.is_available():
            dist.init_process_group(backend="nccl")

        if torch.cuda.is_available():
            lwsize = torch.cuda.device_count()  # local world size - per node
            gwsize = dist.get_world_size()  # global world size - per run
            grank = dist.get_rank()  # global rank - assign per run
            lrank = dist.get_rank() % lwsize  # local rank - assign per node
        else:
            gwsize = 1
            grank = 0
            lrank = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu", lrank)
        if torch.cuda.is_available():
            torch.cuda.set_device(lrank)

        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)

        train_dataloader = _distributed_dataloader(train_dataloader, gwsize, grank)
        if validation_dataloader is not None:
            validation_dataloader = _distributed_dataloader(
                validation_dataloader, gwsize, grank
            )

        try:
            func(model, train_dataloader, validation_dataloader, device, *args, **kwargs)
        finally:
            if torch.cuda.is_available():
                dist.barrier()
                dist.destroy_process_group()

    return dist_train
