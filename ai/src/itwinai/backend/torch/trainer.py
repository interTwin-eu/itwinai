"""Provides training logic for PyTorch models via Trainer classes."""

from typing import Optional, Dict, Union, Callable, Tuple, Type, List, Any
import time
import os
import sys
import numpy as np

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from ..components import Trainer, Logger
from .utils import seed_worker, save_state
from .types import (
    TorchDistributedBackend,
    TorchDistributedStrategy,
    TorchLoss, TorchOptimizer,
    Loss, LrScheduler
)
from .types import TorchDistributedStrategy as StrategyT
from .types import TorchDistributedBackend as BackendT
from .loggers import BaseLogger
from ...utils import dynamically_import_class
from ...cluster import ClusterEnvironment
from ._utils import clear_key


def preproc_dataloader(dataloader: DataLoader, gwsize, grank):
    """Makes a Dataloader distributed."""
    sampler = DistributedSampler(
        dataloader.dataset,
        num_replicas=gwsize,
        rank=grank,
        shuffle=True
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
        pin_memory_device=dataloader.pin_memory_device
    )


def distributed(func):
    """The decorated function must have a standard signature.
    Its first arguments must be:
    model, train_dataloader, validation_dataloader, device (in this order).

    Additional args or kwargs are allowed consistently with the signature
    of the decorated function.
    """
    def dist_train(
            model, train_dataloader, validation_dataloader=None, device='cpu',
            *args, **kwargs
    ):
        if torch.cuda.is_available():
            dist.init_process_group(backend='nccl')

        if torch.cuda.is_available():
            lwsize = torch.cuda.device_count()  # local world size - per node
            gwsize = dist.get_world_size()     # global world size - per run
            grank = dist.get_rank()            # global rank - assign per run
            lrank = dist.get_rank() % lwsize   # local rank - assign per node
        else:
            gwsize = 1
            grank = 0
            lrank = 0

        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu', lrank)
        if torch.cuda.is_available():
            torch.cuda.set_device(lrank)

        model = model.to(device)
        model = DDP(model, device_ids=[device], output_device=device)

        train_dataloader = preproc_dataloader(train_dataloader, gwsize, grank)
        if validation_dataloader is not None:
            validation_dataloader = preproc_dataloader(
                validation_dataloader, gwsize, grank)

        try:
            func(model, train_dataloader, validation_dataloader, device,
                 *args, **kwargs)
        finally:
            if torch.cuda.is_available():
                dist.barrier()
                dist.destroy_process_group()
    return dist_train


class TorchTrainer(Trainer):
    """
    Torch trainer for optionally distributed data-parallel (DDP) workload.
    Assumes to be executed in a SLURM cluster with torchrun. Use the torch
    elastic version of DDP:
    https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun

    Args:
        model (nn.Module): neural network instance.
        loss (Loss): torch loss function instance.
        optimizer (Optimizer): torch optimizer instance.
        epochs (int, optional): number of training epochs. Defaults to 1.
        strategy (Optional[TorchDistributedStrategy], optional): distributed
            strategy. Defaults to StrategyT.NONE.value.
        backend (TorchDistributedBackend, optional): computing backend.
            Defaults to BackendT.NCCL.value.
        shuffle_dataset (bool, optional): whether shuffle dataset before
            sampling batches from dataloader. Defaults to False.
        use_cuda (bool, optional): whether to use GPU. Defaults to True.
        benchrun (bool, optional): sets up a debug run. Defaults to False.
        testrun (bool, optional): deterministic training seeding everything.
            Defaults to False.
        seed (Optional[int], optional): random seed. Defaults to None.
        logger (Optional[List[Logger]], optional): logger. Defaults to None.
        checkpoint_every (int, optional): how often (epochs) to checkpoint the
            best model. Defaults to 10.

    Raises:
        RuntimeError: When trying to use DDP without CUDA support.
        NotImplementedError: when trying to use a strategy different from the
            ones provided by TorchDistributedStrategy.
    """

    model: nn.Module = None
    loss: Loss = None
    optimizer: Optimizer = None
    lr_scheduler = None
    strategy: StrategyT = StrategyT.NONE.value
    backend: BackendT = BackendT.NCCL.value
    train_dataloader: DataLoader = None
    validation_dataloader: DataLoader = None

    def __init__(
        self,
        model: nn.Module,
        loss: Loss,
        optimizer_class: str,
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_class: Optional[str] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        epochs: int = 1,
        strategy: str = StrategyT.NONE.value,
        backend: str = BackendT.NCCL.value,
        shuffle_dataset: bool = False,
        use_cuda: bool = True,
        benchrun: bool = False,
        testrun: bool = False,
        seed: Optional[int] = None,
        logger: Optional[List[Logger]] = None,
        checkpoint_every: int = 10
    ) -> None:
        """Sets up the distributed backend and loggers.
        Makes the model a DDP model.
        """
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.testrun = testrun
        self.seed = seed
        self.strategy = strategy
        self.backend = backend
        self.shuffle_dataset = shuffle_dataset
        self.use_cuda = use_cuda
        self.benchrun = benchrun
        # Checkpoint every n epochs
        self.checkpoint_every = checkpoint_every

        # Optimizer and scheduler
        optim_class = dynamically_import_class(optimizer_class)
        optimizer_kwargs = (
            optimizer_kwargs if optimizer_kwargs is not None else {})
        self.optimizer = optim_class(
            self.model.parameters(), **optimizer_kwargs
        )
        if lr_scheduler_class is not None:
            scheduler_class = dynamically_import_class(lr_scheduler_class)
            lr_scheduler_kwargs = (
                lr_scheduler_kwargs if lr_scheduler_kwargs is not None else {}
            )
            self.lr_scheduler = scheduler_class(
                self.optimizer, **lr_scheduler_kwargs)

        self.cuda = self.use_cuda and torch.cuda.is_available()

        # Init distributed backend
        if self.strategy is not None:
            dist.init_process_group(backend=self.backend)

        # get job rank info - rank==0 master gpu
        if torch.cuda.is_available():
            # local world size - per node
            self.lwsize = torch.cuda.device_count() if self.cuda else 0
            # global world size - per run
            self.gwsize = dist.get_world_size()
            # global rank - assign per run
            self.grank = dist.get_rank()
            # local rank - assign per node
            self.lrank = dist.get_rank() % self.lwsize
        else:
            self.gwsize = 1
            self.grank = 0
            self.lrank = 0

        # Encapsulate the model on the GPU assigned to the current process
        self.device = torch.device(
            'cuda' if self.cuda and torch.cuda.is_available() else 'cpu',
            self.lrank
        )
        if self.cuda:
            torch.cuda.set_device(self.lrank)

        if self.testrun:
            # Deterministic testrun
            torch.manual_seed(self.seed)
            self.g = torch.Generator()
            self.g.manual_seed(self.seed)
            if self.cuda:
                torch.cuda.manual_seed(self.seed)

        self.model = self.model.to(self.device)
        # Create distributed model
        if self.strategy == StrategyT.NONE.value:
            pass
        elif self.strategy == StrategyT.DDP.value:
            if not self.cuda:
                raise RuntimeError(
                    "Cannot use torch distributed data parallel without CUDA."
                )
            self.model = DDP(
                self.model,
                device_ids=[self.device],
                output_device=self.device
            )
        else:
            raise NotImplementedError("Only DDP strategy is implemented.")

        self.logger = (
            logger if logger is not None
            else BaseLogger(create_new=self.grank == 0)
        )

    @property
    def backend(self) -> str:
        return self._backend

    @backend.setter
    def backend(self, backend_name: str) -> None:
        if backend_name not in BackendT:
            raise ValueError(
                "Unrecognized 'backend' field. Allowed values "
                f"are: {BackendT.list()}. Received '{backend_name}'")
        self._backend = backend_name

    @property
    def strategy(self) -> Optional[str]:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_name) -> None:
        if strategy_name not in StrategyT:
            raise ValueError(
                "Unrecognized 'strategy' field. Allowed values "
                f"are: {StrategyT.list()}. Received '{strategy_name}'")
        self._strategy = strategy_name

    def setup(self, args) -> None:
        pass

    def execute(self, args):
        train_dataloader, validation_dataloader = args
        return self._train(train_dataloader, validation_dataloader)

    def _train(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None,
    ):

        # Dataloaders
        self.train_dataloader = self._preproc_dataloader(train_dataloader)
        if validation_dataloader is not None:
            self.validation_dataloader = self._preproc_dataloader(
                validation_dataloader
            )
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        return self.train()

    def _preproc_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Make dataloader distributed if using distributed training strategy.
        TODO: improve using wrapper: 
        https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143?page=2

        Args:
            dataloader (DataLoader): some torch DataLoader instance.
        """
        generator = self.g if self.testrun else dataloader.generator
        worker_init_fn = (
            seed_worker if dataloader.worker_init_fn is None
            else dataloader.worker_init_fn
        )

        if self.strategy is None:
            # No distributed strategy used.
            return DataLoader(
                dataloader.dataset,
                shuffle=self.shuffle_dataset,
                batch_size=dataloader.batch_size,
                sampler=dataloader.sampler,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
                pin_memory=dataloader.pin_memory,
                drop_last=dataloader.drop_last,
                timeout=dataloader.timeout,
                worker_init_fn=worker_init_fn,
                multiprocessing_context=dataloader.multiprocessing_context,
                generator=generator,
                prefetch_factor=dataloader.prefetch_factor,
                persistent_workers=dataloader.persistent_workers,
                pin_memory_device=dataloader.pin_memory_device
            )
        else:
            sampler = DistributedSampler(
                dataloader.dataset,
                num_replicas=self.gwsize,
                rank=self.grank,
                shuffle=self.shuffle_dataset
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
            worker_init_fn=worker_init_fn,
            multiprocessing_context=dataloader.multiprocessing_context,
            generator=generator,
            prefetch_factor=dataloader.prefetch_factor,
            persistent_workers=dataloader.persistent_workers,
            pin_memory_device=dataloader.pin_memory_device
        )

    def training_step(self, batch, batch_idx) -> Loss:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        pred_y = self.model(x)
        return self.loss(pred_y, y)

    def validation_step(self, batch, batch_idx) -> Loss:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        pred_y = self.model(x)
        return self.loss(pred_y, y)

    def training_epoch(self, epoch_idx) -> Loss:
        self.model.train()
        train_losses = []
        # TODO: use tqdm
        for tr_b_idx, train_batch in enumerate(self.train_dataloader):
            loss = self.training_step(
                batch=train_batch,
                batch_idx=tr_b_idx
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss)
        avg_loss = torch.mean(torch.stack(train_losses)).detach().cpu()
        print(f"Avg train loss: {avg_loss}")
        return avg_loss

    def validation_epoch(self, epoch_idx) -> Loss:
        if self.validation_dataloader is not None:
            self.model.eval()
            validation_losses = []
            # TODO: use tqdm
            for val_b_idx, val_batch in enumerate(self.validation_dataloader):
                loss = self.validation_step(
                    batch=val_batch,
                    batch_idx=val_b_idx
                )
                validation_losses.append(loss)
            avg_loss = torch.mean(
                torch.stack(validation_losses)
            ).detach().cpu()
            print(f"Avg validation loss: {avg_loss}")
            return avg_loss

    def train(self):

        if self.optimizer is None:
            raise ValueError("Undefined optimizer!")

        if self.loss is None:
            raise ValueError("Undefined loss function!")

        st = time.time()

        # Resume state
        start_epoch = 1
        best_loss = np.Inf
        res_name = os.path.join(self.logger.run_path, 'checkpoint.pth.tar')
        if os.path.isfile(res_name) and not self.benchrun:
            try:
                if torch.cuda.is_available():
                    dist.barrier()
                    # Map model to be loaded to specified single gpu.
                    loc = {'cuda:%d' % 0: 'cuda:%d' % self.lrank} if self.cuda else {
                        'cpu:%d' % 0: 'cpu:%d' % self.lrank}
                    checkpoint = torch.load(res_name, map_location=loc)
                else:
                    checkpoint = torch.load(res_name, map_location='cpu')
                start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if torch.cuda.is_available():
                    if self.grank == 0:
                        print(f'WARNING: restarting from {start_epoch} epoch')
                else:
                    print(f'WARNING: restarting from {start_epoch} epoch')
            except:
                if torch.cuda.is_available():
                    if self.grank == 0:
                        print('WARNING: restart file cannot be loaded, restarting!')
                else:
                    print('WARNING: restart file cannot be loaded, restarting!')

        if start_epoch >= self.epochs + 1:
            if torch.cuda.is_available():
                if self.grank == 0:
                    print('WARNING: given epochs are less than the one in the restart file!\n'
                          'WARNING: SYS.EXIT is issued')
                dist.destroy_process_group()
                sys.exit()
            else:
                print('WARNING: given epochs are less than the one in the restart file!\n'
                      'WARNING: SYS.EXIT is issued')
                sys.exit()

        # start trainin/testing loop
        if self.grank == 0:
            print('TIMER: broadcast:', time.time()-st, 's')
            print(f'\nDEBUG: start training')
            print(f'--------------------------------------------------------')

        et = time.time()
        # TODO use tqdm? For distributed situations could be difficult
        for epoch_idx in range(start_epoch, self.epochs + 1):
            lt = time.time()

            if self.benchrun and epoch_idx == self.epochs:
                # profiling (done on last epoch - slower!)
                with torch.autograd.profiler.profile(use_cuda=self.cuda,
                                                     profile_memory=True) as prof:
                    train_loss = self.training_epoch(epoch_idx=epoch_idx)
            else:
                train_loss = self.training_epoch(epoch_idx=epoch_idx)
            val_loss = self.validation_epoch(epoch_idx=epoch_idx)

            # save first epoch timer
            if epoch_idx == start_epoch:
                first_ep_t = time.time()-lt

            # final epoch
            if epoch_idx + 1 == self.epochs:
                self.train_dataloader.last_epoch = True
                self.validation_dataloader.last_epoch = True

            if self.grank == 0:
                print('TIMER: epoch time:', time.time()-lt, 's')
                if self.benchrun and epoch_idx == self.epochs:
                    print('\n--------------------------------------------------------')
                    print('DEBUG: benchmark of last epoch:\n')
                    what1 = 'cuda' if self.cuda else 'cpu'
                    print(prof.key_averages().table(
                        sort_by='self_'+str(what1)+'_time_total'))

            # save state if found a better state
            ref_loss = val_loss if val_loss is not None else train_loss
            is_best = ref_loss < best_loss
            if epoch_idx % self.checkpoint_every == 0 and not self.benchrun:
                save_state(
                    epoch_idx, self.model, ref_loss, self.optimizer,
                    res_name, self.grank, self.gwsize, is_best
                )
                # reset best_acc
                best_loss = min(ref_loss, best_loss)

        # save final state
        if not self.benchrun:
            save_state(
                epoch_idx, self.model, ref_loss,
                self.optimizer, res_name, self.grank, self.gwsize, True
            )
        if torch.cuda.is_available():
            dist.barrier()

        # some debug
        if self.grank == 0:
            print('\n--------------------------------------------------------')
            print('DEBUG: training results:\n')
            print('TIMER: first epoch time:', first_ep_t, ' s')
            print('TIMER: last epoch time:', time.time()-lt, ' s')
            print('TIMER: average epoch time:',
                  (time.time()-et)/self.epochs, ' s')
            print('TIMER: total epoch time:', time.time()-et, ' s')
            if epoch_idx > 1:
                print('TIMER: total epoch-1 time:',
                      time.time()-et-first_ep_t, ' s')
                print('TIMER: average epoch-1 time:',
                      (time.time()-et-first_ep_t)/(self.epochs-1), ' s')
            if self.benchrun:
                print('TIMER: total epoch-2 time:', lt-first_ep_t, ' s')
                print('TIMER: average epoch-2 time:',
                      (lt-first_ep_t)/(self.epochs-2), ' s')
            print('DEBUG: memory req:', int(torch.cuda.memory_reserved(self.lrank)/1024/1024), 'MB') \
                if self.cuda else 'DEBUG: memory req: - MB'
            print('DEBUG: memory summary:\n\n',
                  torch.cuda.memory_summary(0)) if self.cuda else ''

        if self.grank == 0:
            print(f'TIMER: final time: {time.time()-st} s\n')

        # TODO: use a with?
        self.cleanup()

    def cleanup(self):
        """
        Destroy a given process group, and deinitialize the distributed
        package.
        """
        if torch.cuda.is_available():
            dist.barrier()
            dist.destroy_process_group()


class TorchTrainerMG2(Trainer):
    """
    Torch trainer for optionally distributed data-parallel (DDP) workload.
    Multi-GPU distribution.

    Args:
        model (nn.Module): neural network instance.
        loss (Loss): torch loss function instance.
        optimizer_class (str): path to optimizer class
            (e.g., 'torch.optim.SGD')
        optimizer_kwargs (Optional[Dict], optional): optimizer constructor
            arguments (except from parameters). Defaults to None.
        lr_scheduler_class (Optional[str], optional): path to learning
            rate scheduler class. Defaults to None.
        lr_scheduler_kwargs (Optional[Dict], optional): constructor arguments
            of the learning rate scheduler, except for the optimizer.
            Defaults to None.
        train_dataloader_class (str, optional): train dataloader class path.
            Defaults to 'torch.utils.data.DataLoader'.
        train_dataloader_kwargs (Optional[Dict], optional): constructor
            arguments of the train dataloader, except for the dataset
            instance. Defaults to None.
        validation_dataloader_class (str, optional): validation dataloader
            class path. Defaults to 'torch.utils.data.DataLoader'.
        validation_dataloader_kwargs (Optional[Dict], optional): constructor
            arguments of the validation dataloader, except for the dataset
            instance. If None, it replicates `train_dataloader_kwargs`.
            Defaults to None.
        epochs (int, optional): number of training epochs. Defaults to 1.
        strategy (Optional[TorchDistributedStrategy], optional): distributed
            strategy. Defaults to StrategyT.NONE.value.
        backend (TorchDistributedBackend, optional): computing backend.
            Defaults to BackendT.NCCL.value.
        shuffle_dataset (bool, optional): whether shuffle dataset before
            sampling batches from dataloader. Defaults to False.
        use_cuda (bool, optional): whether to use GPU. Defaults to True.
        benchrun (bool, optional): sets up a debug run. Defaults to False.
        testrun (bool, optional): deterministic training seeding everything.
            Defaults to False.
        seed (Optional[int], optional): random seed. Defaults to None.
        logger (Optional[List[Logger]], optional): logger. Defaults to None.
        checkpoint_every (int, optional): how often (epochs) to checkpoint the
            best model. Defaults to 10.

    Raises:
        RuntimeError: When trying to use DDP without CUDA support.
        NotImplementedError: when trying to use a strategy different from the
            ones provided by TorchDistributedStrategy.
    """

    model: nn.Module = None
    loss: Loss = None
    optimizer: Optimizer = None
    lr_scheduler = None
    strategy: StrategyT = StrategyT.NONE.value
    train_dataset: Dataset
    validation_dataset: Dataset
    train_dataloader: DataLoader = None
    validation_dataloader: DataLoader = None

    def __init__(
        self,
        model: nn.Module,
        loss: Loss,
        optimizer_class: str,
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_class: Optional[str] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        train_dataloader_class: str = 'torch.utils.data.DataLoader',
        train_dataloader_kwargs: Optional[Dict] = None,
        validation_dataloader_class: str = 'torch.utils.data.DataLoader',
        validation_dataloader_kwargs: Optional[Dict] = None,
        epochs: int = 1,
        strategy: str = StrategyT.NONE.value,
        use_cuda: bool = True,
        benchrun: bool = False,
        testrun: bool = False,
        seed: Optional[int] = None,
        logger: Optional[List[Logger]] = None,
        checkpoint_every: int = 10,
        cluster: Optional[ClusterEnvironment] = None
    ) -> None:
        """Sets up the distributed backend and loggers.
        Makes the model a DDP model.
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.testrun = testrun
        self.seed = seed
        self.strategy = strategy
        # TODO: embed use_cuda into the cluster
        self.use_cuda = use_cuda
        self.benchrun = benchrun
        self.cluster = cluster
        # Checkpoint every n epochs
        self.checkpoint_every = checkpoint_every
        self.cuda = self.use_cuda and torch.cuda.is_available()

        # Train and validation dataloaders
        self.train_dataloader_class = dynamically_import_class(
            train_dataloader_class
        )
        self.validation_dataloader_class = dynamically_import_class(
            validation_dataloader_class
        )
        train_dataloader_kwargs = (
            train_dataloader_kwargs
            if train_dataloader_kwargs is not None else {}
        )
        self.train_dataloader_kwargs = clear_key(
            train_dataloader_kwargs, 'train_dataloader_kwargs', 'dataset'
        )
        # If validation_dataloader_kwargs is not given,
        # copy train_dataloader_kwargs
        validation_dataloader_kwargs = (
            validation_dataloader_kwargs if validation_dataloader_kwargs
            is not None else train_dataloader_kwargs
        )
        self.validation_dataloader_kwargs = clear_key(
            validation_dataloader_kwargs, 'validation_dataloader_kwargs',
            'dataset'
        )

        # Optimizer and scheduler
        optim_class = dynamically_import_class(optimizer_class)
        optimizer_kwargs = (
            optimizer_kwargs if optimizer_kwargs is not None else {}
        )
        optimizer_kwargs = clear_key(
            optimizer_kwargs, 'optimizer_kwargs', 'parameters'
        )
        self.optimizer: Optimizer = optim_class(
            self.model.parameters(), **optimizer_kwargs
        )
        if lr_scheduler_class is not None:
            scheduler_class = dynamically_import_class(lr_scheduler_class)
            lr_scheduler_kwargs = (
                lr_scheduler_kwargs if lr_scheduler_kwargs is not None else {}
            )
            lr_scheduler_kwargs = clear_key(
                lr_scheduler_kwargs, 'lr_scheduler_kwargs', 'optimizer'
            )
            self.lr_scheduler: LrScheduler = scheduler_class(
                self.optimizer, **lr_scheduler_kwargs
            )

        # Loggers
        self.logger = (
            logger if logger is not None
            else BaseLogger(create_new=self.cluster.is_main())
        )

    @property
    def strategy(self) -> Optional[str]:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_name) -> None:
        if strategy_name not in StrategyT:
            raise ValueError(
                "Unrecognized 'strategy' field. Allowed values "
                f"are: {StrategyT.list()}. Received '{strategy_name}'")
        self._strategy = strategy_name

    def set_seed(self, seed: Optional[int] = None):
        """Deterministic operations for reproducibility.
        Sets the random seed.

        Args:
            seed (Optional[int], optional): if not None, overrides
                `self.seed`. Defaults to None.
        """
        seed = seed if seed is not None else self.seed
        np.random.seed(seed)
        self.torch_rng = torch.Generator()
        if seed is not None:
            torch.manual_seed(seed)
            self.torch_rng.manual_seed(seed)
            if self.cuda:
                torch.cuda.manual_seed(seed)

    def setup(self, config: Dict) -> Dict:
        return config

    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        model: nn.Module = None,
        optimizer: Optimizer = None,
        lr_scheduler: LrScheduler = None
    ) -> Any:
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        # Update parameters passed for "interactive" use
        if model is not None:
            self.model = model
        if optimizer is not None:
            self.optimizer = optimizer
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler

        # Start training
        if self.cluster.distributed:
            # Make training distributed
            return mp.spawn(self._train, nprocs=self.cluster.ngpus_per_node)
        else:
            return self._train(0)

    def _train(
        self,
        worker_id: int
    ):
        # Each worker has a different deterministic seed
        # NB: 'worker' = replica of the training function
        worker_seed = (
            self.seed + worker_id if self.seed is not None else self.seed
        )
        self.set_seed(worker_seed)

        # Instantiate dataloaders
        self.train_dataloader = self._instantiate_dataloader(
            dataloader_class=self.train_dataloader_class,
            dataset=self.train_dataset,
            init_kwargs=self.train_dataloader_kwargs
        )
        if self.validation_dataset is not None:
            self.validation_dataloader = self._instantiate_dataloader(
                dataloader_class=self.validation_dataloader_class,
                dataset=self.validation_dataset,
                init_kwargs=self.validation_dataloader_kwargs
            )

        # Launch effective training:

        # Single worker case
        if not self.cluster.distributed:
            with self.cluster.init_dist_gpu(worker_id) as device:
                self.device: torch.device = device
                self.model = self.model.to(self.device)
                return self.train()

        # Init / connect to distributed backend
        with self.cluster.init_dist_gpu(worker_id) as device:
            self.device: torch.device = device
            self._distribute_model()
            train_result = self.train()

        return train_result

    def _instantiate_dataloader(
        self,
        dataloader_class: Type,
        dataset: Dataset,
        init_kwargs: Dict
    ) -> DataLoader:
        """Make dataloader distributed if using distributed training strategy.

        Args:
            dataloader_class (Type): some torch DataLoader type.
            dataset (Dataset): torch dataset instance.
            init_kwargs (Dict): constructor args.
        """
        init_kwargs['generator'] = init_kwargs.get(
            'generator', self.torch_rng
        )
        init_kwargs['worker_init_fn'] = init_kwargs.get(
            'worker_init_fn', seed_worker
        )

        if self.strategy == StrategyT.DDP.value and self.cluster.distributed:
            sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=self.cluster.world_size,
                rank=self.cluster.rank,
                shuffle=init_kwargs.get(
                    'shuffle', False
                )
            )
            # Overwrite existing sampler, if given.
            # TODO: improve using wrapper:
            # https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143?page=2
            init_kwargs['sampler'] = sampler
            if init_kwargs.get('shuffle') is not None:
                # sampler option is mutually exclusive with shuffle
                del init_kwargs['shuffle']

        return dataloader_class(dataset, **init_kwargs)

    def _distribute_model(self):
        if self.cluster.distributed:
            # Distribute model
            self.model = self.model.to(self.device)
            if self.strategy == StrategyT.NONE.value:
                print(
                    "A GPU cluster is available but no distributed "
                    "strategy was given... Falling back to single worker...")
                if not self.cluster.is_main():
                    # Use only GPU:0 for single worker
                    sys.exit(0)
            elif self.strategy == StrategyT.DDP.value:
                self.model = DDP(
                    self.model,
                    device_ids=[self.device.index],
                    output_device=self.device
                )
            else:
                raise NotImplementedError("Only DDP strategy is implemented.")
        else:
            raise RuntimeError(
                "Trying to distribute a model when a "
                "distributed cluster is not available."
            )

    def training_step(self, batch, batch_idx) -> Loss:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        pred_y = self.model(x)
        return self.loss(pred_y, y)

    def validation_step(self, batch, batch_idx) -> Loss:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        pred_y = self.model(x)
        return self.loss(pred_y, y)

    def training_epoch(self, epoch_idx) -> Loss:
        self.model.train()
        train_losses = []
        # TODO: use tqdm
        for tr_b_idx, train_batch in enumerate(self.train_dataloader):
            loss = self.training_step(
                batch=train_batch,
                batch_idx=tr_b_idx
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss)
        avg_loss = torch.mean(torch.stack(train_losses)).detach().cpu()
        print(f"Avg train loss: {avg_loss}")
        return avg_loss

    def validation_epoch(self, epoch_idx) -> Loss:
        if self.validation_dataloader is not None:
            self.model.eval()
            validation_losses = []
            # TODO: use tqdm
            for val_b_idx, val_batch in enumerate(self.validation_dataloader):
                loss = self.validation_step(
                    batch=val_batch,
                    batch_idx=val_b_idx
                )
                validation_losses.append(loss)
            avg_loss = torch.mean(
                torch.stack(validation_losses)
            ).detach().cpu()
            print(f"Avg validation loss: {avg_loss}")
            return avg_loss

    def train(self):

        if self.optimizer is None:
            raise ValueError("Undefined optimizer!")

        if self.loss is None:
            raise ValueError("Undefined loss function!")

        st = time.time()

        # Resume state
        start_epoch = 1
        best_loss = np.Inf
        res_name = os.path.join(self.logger.run_path, 'checkpoint.pth.tar')
        if os.path.isfile(res_name) and not self.benchrun:
            try:
                if torch.cuda.is_available() and self.cluster.distributed:
                    dist.barrier()
                    # Map model to be loaded to specified single gpu.
                    loc = {'cuda:%d' % 0: 'cuda:%d' % self.lrank} if self.cuda else {
                        'cpu:%d' % 0: 'cpu:%d' % self.lrank}
                    checkpoint = torch.load(res_name, map_location=loc)
                else:
                    checkpoint = torch.load(res_name, map_location='cpu')
                start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if torch.cuda.is_available():
                    if self.cluster.rank == 0:
                        print(f'WARNING: restarting from {start_epoch} epoch')
                else:
                    print(f'WARNING: restarting from {start_epoch} epoch')
            except:
                if torch.cuda.is_available():
                    if self.cluster.rank == 0:
                        print('WARNING: restart file cannot be loaded, restarting!')
                else:
                    print('WARNING: restart file cannot be loaded, restarting!')

        if start_epoch >= self.epochs + 1:
            if torch.cuda.is_available() and self.cluster.distributed:
                if self.cluster.rank == 0:
                    print('WARNING: given epochs are less than the one in the restart file!\n'
                          'WARNING: SYS.EXIT is issued')
                dist.destroy_process_group()
                sys.exit()
            else:
                print('WARNING: given epochs are less than the one in the restart file!\n'
                      'WARNING: SYS.EXIT is issued')
                sys.exit()

        # start trainin/testing loop
        if self.cluster.is_main():
            print('TIMER: broadcast:', time.time()-st, 's')
            print(f'\nDEBUG: start training')
            print(f'--------------------------------------------------------')

        et = time.time()
        # TODO use tqdm? For distributed situations could be difficult
        for epoch_idx in range(start_epoch, self.epochs + 1):
            lt = time.time()

            if self.benchrun and epoch_idx == self.epochs:
                # profiling (done on last epoch - slower!)
                with torch.autograd.profiler.profile(use_cuda=self.cuda,
                                                     profile_memory=True) as prof:
                    train_loss = self.training_epoch(epoch_idx=epoch_idx)
            else:
                train_loss = self.training_epoch(epoch_idx=epoch_idx)
            val_loss = self.validation_epoch(epoch_idx=epoch_idx)

            # save first epoch timer
            if epoch_idx == start_epoch:
                first_ep_t = time.time()-lt

            # final epoch
            if epoch_idx + 1 == self.epochs:
                self.train_dataloader.last_epoch = True
                self.validation_dataloader.last_epoch = True

            if self.cluster.rank == 0:
                print('TIMER: epoch time:', time.time()-lt, 's')
                if self.benchrun and epoch_idx == self.epochs:
                    print('\n--------------------------------------------------------')
                    print('DEBUG: benchmark of last epoch:\n')
                    what1 = 'cuda' if self.cuda else 'cpu'
                    print(prof.key_averages().table(
                        sort_by='self_'+str(what1)+'_time_total'))

            # save state if found a better state
            ref_loss = val_loss if val_loss is not None else train_loss
            is_best = ref_loss < best_loss
            if epoch_idx % self.checkpoint_every == 0 and not self.benchrun:
                save_state(
                    epoch_idx, self.model, ref_loss, self.optimizer,
                    res_name, self.cluster.rank, self.cluster.world_size, is_best,
                    distributed=self.cluster.distributed
                )
                # reset best_acc
                best_loss = min(ref_loss, best_loss)

        # save final state
        if not self.benchrun:
            save_state(
                epoch_idx, self.model, ref_loss,
                self.optimizer, res_name, self.cluster.rank, self.cluster.world_size, True,
                distributed=self.cluster.distributed
            )
        if torch.cuda.is_available() and self.cluster.distributed:
            dist.barrier()

        # some debug
        if self.cluster.rank == 0:
            print('\n--------------------------------------------------------')
            print('DEBUG: training results:\n')
            print('TIMER: first epoch time:', first_ep_t, ' s')
            print('TIMER: last epoch time:', time.time()-lt, ' s')
            print('TIMER: average epoch time:',
                  (time.time()-et)/self.epochs, ' s')
            print('TIMER: total epoch time:', time.time()-et, ' s')
            if epoch_idx > 1:
                print('TIMER: total epoch-1 time:',
                      time.time()-et-first_ep_t, ' s')
                print('TIMER: average epoch-1 time:',
                      (time.time()-et-first_ep_t)/(self.epochs-1), ' s')
            if self.benchrun:
                print('TIMER: total epoch-2 time:', lt-first_ep_t, ' s')
                print('TIMER: average epoch-2 time:',
                      (lt-first_ep_t)/(self.epochs-2), ' s')
            # print('DEBUG: memory req:', int(torch.cuda.memory_reserved(self.lrank)/1024/1024), 'MB') \
            #     if self.cuda and self.cluster.distributed else 'DEBUG: memory req: - MB'
            print('DEBUG: memory summary:\n\n',
                  torch.cuda.memory_summary(0)) if self.cuda else ''

        if self.cluster.rank == 0:
            print(f'TIMER: final time: {time.time()-st} s\n')


class TorchTrainerMG(Trainer):
    """
    Torch trainer for optionally distributed data-parallel (DDP) workload.
    Multi-GPU distribution.

    Args:
        model (nn.Module): neural network instance.
        loss (Loss): torch loss function instance.
        optimizer (Optimizer): torch optimizer instance.
        epochs (int, optional): number of training epochs. Defaults to 1.
        strategy (Optional[TorchDistributedStrategy], optional): distributed
            strategy. Defaults to StrategyT.NONE.value.
        backend (TorchDistributedBackend, optional): computing backend.
            Defaults to BackendT.NCCL.value.
        shuffle_dataset (bool, optional): whether shuffle dataset before
            sampling batches from dataloader. Defaults to False.
        use_cuda (bool, optional): whether to use GPU. Defaults to True.
        benchrun (bool, optional): sets up a debug run. Defaults to False.
        testrun (bool, optional): deterministic training seeding everything.
            Defaults to False.
        seed (Optional[int], optional): random seed. Defaults to None.
        logger (Optional[List[Logger]], optional): logger. Defaults to None.
        checkpoint_every (int, optional): how often (epochs) to checkpoint the
            best model. Defaults to 10.

    Raises:
        RuntimeError: When trying to use DDP without CUDA support.
        NotImplementedError: when trying to use a strategy different from the
            ones provided by TorchDistributedStrategy.
    """

    model: nn.Module = None
    loss: Loss = None
    optimizer: Optimizer = None
    lr_scheduler = None
    strategy: StrategyT = StrategyT.NONE.value
    train_dataloader: DataLoader = None
    validation_dataloader: DataLoader = None

    def __init__(
        self,
        model: nn.Module,
        loss: Loss,
        optimizer_class: str,
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_class: Optional[str] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        epochs: int = 1,
        strategy: str = StrategyT.NONE.value,
        shuffle_dataset: bool = False,
        use_cuda: bool = True,
        benchrun: bool = False,
        testrun: bool = False,
        seed: Optional[int] = None,
        logger: Optional[List[Logger]] = None,
        checkpoint_every: int = 10,
        cluster: Optional[ClusterEnvironment] = None
    ) -> None:
        """Sets up the distributed backend and loggers.
        Makes the model a DDP model.
        """
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.testrun = testrun
        self.seed = seed
        self.strategy = strategy
        self.shuffle_dataset = shuffle_dataset
        self.use_cuda = use_cuda
        self.benchrun = benchrun
        self.cluster = cluster
        # Checkpoint every n epochs
        self.checkpoint_every = checkpoint_every

        # Optimizer and scheduler
        optim_class = dynamically_import_class(optimizer_class)
        optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.optimizer = optim_class(
            self.model.parameters(), **optimizer_kwargs
        )
        if lr_scheduler_class is not None:
            scheduler_class = dynamically_import_class(lr_scheduler_class)
            lr_scheduler_kwargs = (
                lr_scheduler_kwargs if lr_scheduler_kwargs is not None else {}
            )
            self.lr_scheduler = scheduler_class(
                self.optimizer, **lr_scheduler_kwargs)

        self.cuda = self.use_cuda and torch.cuda.is_available()

        # if self.testrun:
        #     # Deterministic testrun
        #     torch.manual_seed(self.seed)
        #     self.g = torch.Generator()
        #     self.g.manual_seed(self.seed)
        #     if self.cuda:
        #         torch.cuda.manual_seed(self.seed)

        self.logger = (
            logger if logger is not None
            else BaseLogger(create_new=self.cluster.is_main())
        )

    @property
    def strategy(self) -> Optional[str]:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_name) -> None:
        if strategy_name not in StrategyT:
            raise ValueError(
                "Unrecognized 'strategy' field. Allowed values "
                f"are: {StrategyT.list()}. Received '{strategy_name}'")
        self._strategy = strategy_name

    def setup(self, config: Dict) -> Dict:
        return config

    def execute(self, args) -> Any:
        self.train_dataloader, self.validation_dataloader = args
        if self.cluster.ngpus_per_node > 1:
            # Make training distributed
            return mp.spawn(self._train, nprocs=self.cluster.ngpus_per_node)
        else:
            return self._train(0)

    def _train(
        self,
        worker_id: int
    ):
        if not self.cluster.distributed:
            with self.cluster.init_dist_gpu(worker_id) as device:
                self.device: torch.device = device
                self.model = self.model.to(self.device)
                return self.train()

        # Init / connect to distributed backend
        with self.cluster.init_dist_gpu(worker_id) as device:
            self.device: torch.device = device

            # Distribute dataloaders (if needed)
            self.train_dataloader = self._preproc_dataloader(
                self.train_dataloader)
            if self.validation_dataloader is not None:
                self.validation_dataloader = self._preproc_dataloader(
                    self.validation_dataloader
                )

            # Distribute model
            self.model = self.model.to(self.device)
            if self.strategy == StrategyT.NONE.value:
                print(
                    "A GPU cluster is available but no distributed "
                    "strategy was given... Falling back to single worker...")
                if not self.cluster.is_main():
                    # Use only GPU:0 for single worker
                    return
            elif self.strategy == StrategyT.DDP.value:
                self.model = DDP(
                    self.model,
                    device_ids=[self.device.index],
                    output_device=self.device
                )
            else:
                raise NotImplementedError("Only DDP strategy is implemented.")

            train_result = self.train()

        return train_result

    def _preproc_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Make dataloader distributed if using distributed training strategy.
        TODO: improve using wrapper: 
        https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143?page=2

        Args:
            dataloader (DataLoader): some torch DataLoader instance.
        """
        generator = self.g if self.testrun else dataloader.generator
        worker_init_fn = (
            seed_worker if dataloader.worker_init_fn is None
            else dataloader.worker_init_fn
        )

        if self.strategy is None:
            # No distributed strategy used.
            return DataLoader(
                dataloader.dataset,
                shuffle=self.shuffle_dataset,
                batch_size=dataloader.batch_size,
                sampler=dataloader.sampler,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
                pin_memory=dataloader.pin_memory,
                drop_last=dataloader.drop_last,
                timeout=dataloader.timeout,
                worker_init_fn=worker_init_fn,
                multiprocessing_context=dataloader.multiprocessing_context,
                generator=generator,
                prefetch_factor=dataloader.prefetch_factor,
                persistent_workers=dataloader.persistent_workers,
                pin_memory_device=dataloader.pin_memory_device
            )
        else:
            sampler = DistributedSampler(
                dataloader.dataset,
                num_replicas=self.cluster.world_size,
                rank=self.cluster.rank,
                shuffle=self.shuffle_dataset
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
            worker_init_fn=worker_init_fn,
            multiprocessing_context=dataloader.multiprocessing_context,
            generator=generator,
            prefetch_factor=dataloader.prefetch_factor,
            persistent_workers=dataloader.persistent_workers,
            pin_memory_device=dataloader.pin_memory_device
        )

    def training_step(self, batch, batch_idx) -> Loss:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        pred_y = self.model(x)
        return self.loss(pred_y, y)

    def validation_step(self, batch, batch_idx) -> Loss:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        pred_y = self.model(x)
        return self.loss(pred_y, y)

    def training_epoch(self, epoch_idx) -> Loss:
        self.model.train()
        train_losses = []
        # TODO: use tqdm
        for tr_b_idx, train_batch in enumerate(self.train_dataloader):
            loss = self.training_step(
                batch=train_batch,
                batch_idx=tr_b_idx
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss)
        avg_loss = torch.mean(torch.stack(train_losses)).detach().cpu()
        print(f"Avg train loss: {avg_loss}")
        return avg_loss

    def validation_epoch(self, epoch_idx) -> Loss:
        if self.validation_dataloader is not None:
            self.model.eval()
            validation_losses = []
            # TODO: use tqdm
            for val_b_idx, val_batch in enumerate(self.validation_dataloader):
                loss = self.validation_step(
                    batch=val_batch,
                    batch_idx=val_b_idx
                )
                validation_losses.append(loss)
            avg_loss = torch.mean(
                torch.stack(validation_losses)
            ).detach().cpu()
            print(f"Avg validation loss: {avg_loss}")
            return avg_loss

    def train(self):

        if self.optimizer is None:
            raise ValueError("Undefined optimizer!")

        if self.loss is None:
            raise ValueError("Undefined loss function!")

        st = time.time()

        # Resume state
        start_epoch = 1
        best_loss = np.Inf
        res_name = os.path.join(self.logger.run_path, 'checkpoint.pth.tar')
        if os.path.isfile(res_name) and not self.benchrun:
            try:
                if torch.cuda.is_available() and self.cluster.distributed:
                    dist.barrier()
                    # Map model to be loaded to specified single gpu.
                    loc = {'cuda:%d' % 0: 'cuda:%d' % self.lrank} if self.cuda else {
                        'cpu:%d' % 0: 'cpu:%d' % self.lrank}
                    checkpoint = torch.load(res_name, map_location=loc)
                else:
                    checkpoint = torch.load(res_name, map_location='cpu')
                start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if torch.cuda.is_available():
                    if self.cluster.rank == 0:
                        print(f'WARNING: restarting from {start_epoch} epoch')
                else:
                    print(f'WARNING: restarting from {start_epoch} epoch')
            except:
                if torch.cuda.is_available():
                    if self.cluster.rank == 0:
                        print('WARNING: restart file cannot be loaded, restarting!')
                else:
                    print('WARNING: restart file cannot be loaded, restarting!')

        if start_epoch >= self.epochs + 1:
            if torch.cuda.is_available() and self.cluster.distributed:
                if self.cluster.rank == 0:
                    print('WARNING: given epochs are less than the one in the restart file!\n'
                          'WARNING: SYS.EXIT is issued')
                dist.destroy_process_group()
                sys.exit()
            else:
                print('WARNING: given epochs are less than the one in the restart file!\n'
                      'WARNING: SYS.EXIT is issued')
                sys.exit()

        # start trainin/testing loop
        if self.cluster.is_main():
            print('TIMER: broadcast:', time.time()-st, 's')
            print(f'\nDEBUG: start training')
            print(f'--------------------------------------------------------')

        et = time.time()
        # TODO use tqdm? For distributed situations could be difficult
        for epoch_idx in range(start_epoch, self.epochs + 1):
            lt = time.time()

            if self.benchrun and epoch_idx == self.epochs:
                # profiling (done on last epoch - slower!)
                with torch.autograd.profiler.profile(use_cuda=self.cuda,
                                                     profile_memory=True) as prof:
                    train_loss = self.training_epoch(epoch_idx=epoch_idx)
            else:
                train_loss = self.training_epoch(epoch_idx=epoch_idx)
            val_loss = self.validation_epoch(epoch_idx=epoch_idx)

            # save first epoch timer
            if epoch_idx == start_epoch:
                first_ep_t = time.time()-lt

            # final epoch
            if epoch_idx + 1 == self.epochs:
                self.train_dataloader.last_epoch = True
                self.validation_dataloader.last_epoch = True

            if self.cluster.rank == 0:
                print('TIMER: epoch time:', time.time()-lt, 's')
                if self.benchrun and epoch_idx == self.epochs:
                    print('\n--------------------------------------------------------')
                    print('DEBUG: benchmark of last epoch:\n')
                    what1 = 'cuda' if self.cuda else 'cpu'
                    print(prof.key_averages().table(
                        sort_by='self_'+str(what1)+'_time_total'))

            # save state if found a better state
            ref_loss = val_loss if val_loss is not None else train_loss
            is_best = ref_loss < best_loss
            if epoch_idx % self.checkpoint_every == 0 and not self.benchrun:
                save_state(
                    epoch_idx, self.model, ref_loss, self.optimizer,
                    res_name, self.cluster.rank, self.cluster.world_size, is_best,
                    distributed=self.cluster.distributed
                )
                # reset best_acc
                best_loss = min(ref_loss, best_loss)

        # save final state
        if not self.benchrun:
            save_state(
                epoch_idx, self.model, ref_loss,
                self.optimizer, res_name, self.cluster.rank, self.cluster.world_size, True,
                distributed=self.cluster.distributed
            )
        if torch.cuda.is_available() and self.cluster.distributed:
            dist.barrier()

        # some debug
        if self.cluster.rank == 0:
            print('\n--------------------------------------------------------')
            print('DEBUG: training results:\n')
            print('TIMER: first epoch time:', first_ep_t, ' s')
            print('TIMER: last epoch time:', time.time()-lt, ' s')
            print('TIMER: average epoch time:',
                  (time.time()-et)/self.epochs, ' s')
            print('TIMER: total epoch time:', time.time()-et, ' s')
            if epoch_idx > 1:
                print('TIMER: total epoch-1 time:',
                      time.time()-et-first_ep_t, ' s')
                print('TIMER: average epoch-1 time:',
                      (time.time()-et-first_ep_t)/(self.epochs-1), ' s')
            if self.benchrun:
                print('TIMER: total epoch-2 time:', lt-first_ep_t, ' s')
                print('TIMER: average epoch-2 time:',
                      (lt-first_ep_t)/(self.epochs-2), ' s')
            # print('DEBUG: memory req:', int(torch.cuda.memory_reserved(self.lrank)/1024/1024), 'MB') \
            #     if self.cuda and self.cluster.distributed else 'DEBUG: memory req: - MB'
            print('DEBUG: memory summary:\n\n',
                  torch.cuda.memory_summary(0)) if self.cuda else ''

        if self.cluster.rank == 0:
            print(f'TIMER: final time: {time.time()-st} s\n')


class TorchTrainer2(Trainer):
    """
    Torch trainer for optionally distributed data-parallel (DDP) workload.
    Assumes to be executed in a SLURM cluster with torchrun. Use the torch
    elastic version of DDP:
    https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun

    TODO: load form config file (how to do with the optimizer?)
    TODO: complete loss function and optimizer defaults
    """

    model: nn.Module = None
    optimizer: Optimizer = None
    _loss: Callable = None
    train_dataloader: DataLoader = None
    validation_dataloader: DataLoader = None
    strategy: TorchDistributedStrategy = None
    backend: TorchDistributedBackend = 'nccl'

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None,
        epochs: int = 1,
        loss: Union[Callable, TorchLoss] = 'MSELoss',
        optimizer: Union[Optimizer, TorchOptimizer] = 'SGD',
        optimizer_kwargs: Optional[Dict] = None,
        testrun: bool = False,
        shuffle_data: bool = False,
        seed: Optional[int] = None,
        log_int: int = 10,
        strategy: Optional[TorchDistributedStrategy] = None,
        backend: TorchDistributedBackend = 'nccl',
        use_cuda: bool = True,
        benchrun: bool = False,
        logger: Optional[List[Logger]] = None,
        checkpoint_every: int = 10
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.loss = loss
        self.testrun = testrun
        self.seed = seed
        self.shuffle_data = shuffle_data
        self.log_int = log_int
        self.strategy = strategy
        self.backend = backend
        self.use_cuda = use_cuda
        self.benchrun = benchrun
        # Checkpoint every n epochs
        self.checkpoint_every = checkpoint_every

        self.cuda = self.use_cuda and torch.cuda.is_available()

        # Init distributed backend
        if self.strategy is not None:
            dist.init_process_group(backend=self.backend)

        # get job rank info - rank==0 master gpu
        if torch.cuda.is_available():
            # local world size - per node
            self.lwsize = torch.cuda.device_count() if self.cuda else 0
            # global world size - per run
            self.gwsize = dist.get_world_size()
            # global rank - assign per run
            self.grank = dist.get_rank()
            # local rank - assign per node
            self.lrank = dist.get_rank() % self.lwsize
        else:
            self.gwsize = 1
            self.grank = 0
            self.lrank = 0

        # encapsulate the model on the GPU assigned to the current process
        self.device = torch.device(
            'cuda' if self.cuda and torch.cuda.is_available() else 'cpu',
            self.lrank
        )
        if self.cuda:
            torch.cuda.set_device(self.lrank)

        if self.testrun:
            # deterministic testrun
            torch.manual_seed(self.seed)
            self.g = torch.Generator()
            self.g.manual_seed(self.seed)
            if self.cuda:
                torch.cuda.manual_seed(self.seed)

        self.model = self.model.to(self.device)
        # Create distributed model
        if self.strategy == TorchDistributedStrategy.NONE.value:
            pass
        elif self.strategy == TorchDistributedStrategy.DDP.value:
            if not self.cuda:
                raise RuntimeError(
                    "Cannot use torch distributed data parallel without CUDA."
                )
            self.model = DDP(
                self.model,
                device_ids=[self.device],
                output_device=self.device
            )
        else:
            raise NotImplementedError("Only DDP strategy is implemented.")

        # Optimizers
        self.optimizer = self.configure_optimizers(
            optim=optimizer, optim_kwargs=optimizer_kwargs)

        # Dataloaders
        self.train_dataloader = self._preproc_dataloader(train_dataloader)
        if validation_dataloader is not None:
            self.validation_dataloader = self._preproc_dataloader(
                validation_dataloader
            )

        self.logger = (
            logger if logger is not None
            else BaseLogger(create_new=self.grank == 0)
        )

    @property
    def backend(self) -> str:
        return self._backend

    @backend.setter
    def backend(self, backend_name: str) -> None:
        if backend_name not in TorchDistributedBackend:
            raise ValueError(
                "Unrecognized 'backend' field. Allowed values "
                f"are: {TorchDistributedBackend.list()}")
        self._backend = backend_name

    @property
    def strategy(self) -> Optional[str]:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_name) -> None:
        if strategy_name not in TorchDistributedStrategy:
            raise ValueError(
                "Unrecognized 'strategy' field. Allowed values "
                f"are: {TorchDistributedStrategy.list()}")
        self._strategy = strategy_name

    @property
    def loss(self) -> Callable:
        return self._loss

    @loss.setter
    def loss(self, loss: Union[Callable, TorchLoss]) -> None:
        if hasattr(loss, '__call__'):
            self._loss = loss
        elif isinstance(loss, str) and loss in TorchLoss:
            self._loss = self._default_loss(loss)
        else:
            raise ValueError(
                "Unrecognized loss type (if you gave a string, it has to be case sensitive).")

    def _default_loss(self, loss: TorchLoss) -> Callable:
        if loss == TorchLoss.L1.value:
            return nn.L1Loss()
        if loss == TorchLoss.MSE.value:
            return nn.MSELoss()
        if loss == TorchLoss.CROSS_ENTROPY.value:
            return nn.CrossEntropyLoss()
        if loss == TorchLoss.NLLLOSS.value:
            return nn.NLLLoss()

        # TODO: support all losses form https://pytorch.org/docs/stable/nn.html#loss-functions
        raise NotImplementedError(
            "Argh! Support for other losses is still missing...")

    def configure_optimizers(
        self,
        optim: Union[Optimizer, TorchOptimizer],
        optim_kwargs: Optional[Dict] = None
    ) -> Optimizer:
        if isinstance(optim, Optimizer):
            # The optimizer is already instantiated
            return optim

        if isinstance(optim, str) and optim in TorchOptimizer:
            # Optimizer has to be instantiated from its name and kwargs
            optimizer_class, def_args = self._default_optimizer_class(optim)
            optim_kwargs = def_args if optim_kwargs is None else def_args.update(
                optim_kwargs)
            return optimizer_class(self.model.parameters(), **optim_kwargs)

        raise ValueError(
            "Unrecognized optimizer type (if you gave a string, "
            "it has to be case sensitive)."
        )

    def _default_optimizer_class(self, optim: TorchOptimizer) -> Tuple[Type, Dict]:
        """
        Returns optimizer class and a default value for its required construnctor args, if any.
        """
        if optim == TorchOptimizer.SGD.value:
            return torch.optim.SGD, dict(lr=1e-3)
        if optim == TorchOptimizer.ADAM.value:
            return torch.optim.Adam, dict()

        # TODO: support all optimizers from https://pytorch.org/docs/stable/optim.html#algorithms
        raise NotImplementedError(
            "Argh! Support for other losses is still missing...")

    def setup(self, args) -> None:
        pass

    def execute(self, *args, **kwargs):
        return self.train(*args, **kwargs)

    def _preproc_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Make dataloader distributed if using distributed training strategy.

        Args:
            dataloader (DataLoader): some torch DataLoader instance.
        """
        generator = self.g if self.testrun else dataloader.generator

        if self.strategy is None:
            # No distributed strategy used.
            return DataLoader(
                dataloader.dataset,
                shuffle=self.shuffle_data,
                batch_size=dataloader.batch_size,
                sampler=dataloader.sampler,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
                pin_memory=dataloader.pin_memory,
                drop_last=dataloader.drop_last,
                timeout=dataloader.timeout,
                worker_init_fn=seed_worker,  # dataloader.worker_init_fn,
                multiprocessing_context=dataloader.multiprocessing_context,
                generator=generator,
                prefetch_factor=dataloader.prefetch_factor,
                persistent_workers=dataloader.persistent_workers,
                pin_memory_device=dataloader.pin_memory_device
            )
        else:
            sampler = DistributedSampler(
                dataloader.dataset,
                num_replicas=self.gwsize,
                rank=self.grank,
                shuffle=self.shuffle_data
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
            generator=generator,
            prefetch_factor=dataloader.prefetch_factor,
            persistent_workers=dataloader.persistent_workers,
            pin_memory_device=dataloader.pin_memory_device
        )

    def training_step(self, batch, batch_idx) -> Loss:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        pred_y = self.model(x)
        return self.loss(pred_y, y)

    def validation_step(self, batch, batch_idx) -> Loss:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        pred_y = self.model(x)
        return self.loss(pred_y, y)

    def training_epoch(self, epoch_idx) -> Loss:
        self.model.train()
        train_losses = []
        # TODO: use tqdm
        for tr_b_idx, train_batch in enumerate(self.train_dataloader):
            loss = self.training_step(
                batch=train_batch,
                batch_idx=tr_b_idx
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss)
        avg_loss = torch.mean(torch.stack(train_losses)).detach().cpu()
        print(f"Avg train loss: {avg_loss}")
        return avg_loss

    def validation_epoch(self, epoch_idx) -> Loss:
        if self.validation_dataloader is not None:
            self.model.eval()
            validation_losses = []
            # TODO: use tqdm
            for val_b_idx, val_batch in enumerate(self.validation_dataloader):
                loss = self.validation_step(
                    batch=val_batch,
                    batch_idx=val_b_idx
                )
                validation_losses.append(loss)
            avg_loss = torch.mean(
                torch.stack(validation_losses)
            ).detach().cpu()
            print(f"Avg validation loss: {avg_loss}")
            return avg_loss

    def train(self):

        if self.optimizer is None:
            raise ValueError("Undefined optimizer!")

        if self.loss is None:
            raise ValueError("Undefined loss function!")

        st = time.time()

        # Resume state
        start_epoch = 1
        best_loss = np.Inf
        res_name = os.path.join(self.logger.run_path, 'checkpoint.pth.tar')
        if os.path.isfile(res_name) and not self.benchrun:
            try:
                if torch.cuda.is_available():
                    dist.barrier()
                    # Map model to be loaded to specified single gpu.
                    loc = {'cuda:%d' % 0: 'cuda:%d' % self.lrank} if self.cuda else {
                        'cpu:%d' % 0: 'cpu:%d' % self.lrank}
                    checkpoint = torch.load(res_name, map_location=loc)
                else:
                    checkpoint = torch.load(res_name, map_location='cpu')
                start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if torch.cuda.is_available():
                    if self.grank == 0:
                        print(f'WARNING: restarting from {start_epoch} epoch')
                else:
                    print(f'WARNING: restarting from {start_epoch} epoch')
            except:
                if torch.cuda.is_available():
                    if self.grank == 0:
                        print('WARNING: restart file cannot be loaded, restarting!')
                else:
                    print('WARNING: restart file cannot be loaded, restarting!')

        if start_epoch >= self.epochs + 1:
            if torch.cuda.is_available():
                if self.grank == 0:
                    print('WARNING: given epochs are less than the one in the restart file!\n'
                          'WARNING: SYS.EXIT is issued')
                dist.destroy_process_group()
                sys.exit()
            else:
                print('WARNING: given epochs are less than the one in the restart file!\n'
                      'WARNING: SYS.EXIT is issued')
                sys.exit()

        # start trainin/testing loop
        if self.grank == 0:
            print('TIMER: broadcast:', time.time()-st, 's')
            print(f'\nDEBUG: start training')
            print(f'--------------------------------------------------------')

        et = time.time()
        # TODO use tqdm? For distributed situations could be difficult
        for epoch_idx in range(start_epoch, self.epochs + 1):
            lt = time.time()

            if self.benchrun and epoch_idx == self.epochs:
                # profiling (done on last epoch - slower!)
                with torch.autograd.profiler.profile(use_cuda=self.cuda,
                                                     profile_memory=True) as prof:
                    train_loss = self.training_epoch(epoch_idx=epoch_idx)
            else:
                train_loss = self.training_epoch(epoch_idx=epoch_idx)
            val_loss = self.validation_epoch(epoch_idx=epoch_idx)

            # save first epoch timer
            if epoch_idx == start_epoch:
                first_ep_t = time.time()-lt

            # final epoch
            if epoch_idx + 1 == self.epochs:
                self.train_dataloader.last_epoch = True
                self.validation_dataloader.last_epoch = True

            if self.grank == 0:
                print('TIMER: epoch time:', time.time()-lt, 's')
                if self.benchrun and epoch_idx == self.epochs:
                    print('\n--------------------------------------------------------')
                    print('DEBUG: benchmark of last epoch:\n')
                    what1 = 'cuda' if self.cuda else 'cpu'
                    print(prof.key_averages().table(
                        sort_by='self_'+str(what1)+'_time_total'))

            # save state if found a better state
            ref_loss = val_loss if val_loss is not None else train_loss
            is_best = ref_loss < best_loss
            if epoch_idx % self.checkpoint_every == 0 and not self.benchrun:
                save_state(
                    epoch_idx, self.model, ref_loss, self.optimizer,
                    res_name, self.grank, self.gwsize, is_best
                )
                # reset best_acc
                best_loss = min(ref_loss, best_loss)

        # save final state
        if not self.benchrun:
            save_state(
                epoch_idx, self.model, ref_loss,
                self.optimizer, res_name, self.grank, self.gwsize, True
            )
        if torch.cuda.is_available():
            dist.barrier()

        # some debug
        if self.grank == 0:
            print('\n--------------------------------------------------------')
            print('DEBUG: training results:\n')
            print('TIMER: first epoch time:', first_ep_t, ' s')
            print('TIMER: last epoch time:', time.time()-lt, ' s')
            print('TIMER: average epoch time:',
                  (time.time()-et)/self.epochs, ' s')
            print('TIMER: total epoch time:', time.time()-et, ' s')
            if epoch_idx > 1:
                print('TIMER: total epoch-1 time:',
                      time.time()-et-first_ep_t, ' s')
                print('TIMER: average epoch-1 time:',
                      (time.time()-et-first_ep_t)/(self.epochs-1), ' s')
            if self.benchrun:
                print('TIMER: total epoch-2 time:', lt-first_ep_t, ' s')
                print('TIMER: average epoch-2 time:',
                      (lt-first_ep_t)/(self.epochs-2), ' s')
            print('DEBUG: memory req:', int(torch.cuda.memory_reserved(self.lrank)/1024/1024), 'MB') \
                if self.cuda else 'DEBUG: memory req: - MB'
            print('DEBUG: memory summary:\n\n',
                  torch.cuda.memory_summary(0)) if self.cuda else ''

        if self.grank == 0:
            print(f'TIMER: final time: {time.time()-st} s\n')

        # TODO: use a with?
        self.cleanup()

    def cleanup(self):
        """
        Destroy a given process group, and deinitialize the distributed
        package.
        """
        if torch.cuda.is_available():
            dist.barrier()
            dist.destroy_process_group()


class TorchTrainer3(Trainer):
    """
    Torch trainer for optionally distributed data-parallel (DDP) workload.
    Assumes to be executed in a SLURM cluster with torchrun. Use the torch
    elastic version of DDP:
    https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun

    TODO: load form config file (how to do with the optimizer?)
    TODO: complete loss function and optimizer defaults
    """

    model: nn.Module = None
    optimizer: Optimizer = None
    _loss: Callable = None
    train_dataloader: DataLoader = None
    validation_dataloader: DataLoader = None
    strategy: TorchDistributedStrategy = None
    backend: TorchDistributedBackend = 'nccl'

    def __init__(
        self,
        epochs: int = 1,
        loss: Loss = None,
        optimizer_class: str = 'torch.optim.SGD',
        optimizer_kwargs: Optional[Dict] = None,
        testrun: bool = False,
        shuffle_data: bool = False,
        seed: Optional[int] = None,
        log_int: int = 10,
        strategy: StrategyT = StrategyT.NONE.value,
        backend: BackendT = BackendT.NCCL.value,
        use_cuda: bool = True,
        benchrun: bool = False,
        logger: Optional[List[Logger]] = None,
        checkpoint_every: int = 10
    ) -> None:
        self.epochs = epochs
        self.loss = loss
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = (
            optimizer_kwargs if optimizer_kwargs is not None
            else dict(lr=1e-3)
        )
        self.testrun = testrun
        self.shuffle_data = shuffle_data
        self.seed = seed
        self.log_int = log_int
        self.strategy = strategy
        self.backend = backend
        self.use_cuda = use_cuda
        self.benchrun = benchrun
        # Checkpoint every n epochs
        self.checkpoint_every = checkpoint_every

        self.cuda = self.use_cuda and torch.cuda.is_available()

        # Init distributed backend
        if self.strategy is not None:
            dist.init_process_group(backend=self.backend)

        # get job rank info - rank==0 master gpu
        if torch.cuda.is_available():
            # local world size - per node
            self.lwsize = torch.cuda.device_count() if self.cuda else 0
            # global world size - per run
            self.gwsize = dist.get_world_size()
            # global rank - assign per run
            self.grank = dist.get_rank()
            # local rank - assign per node
            self.lrank = dist.get_rank() % self.lwsize
        else:
            self.gwsize = 1
            self.grank = 0
            self.lrank = 0

        # encapsulate the model on the GPU assigned to the current process
        self.device = torch.device(
            'cuda' if self.cuda and torch.cuda.is_available() else 'cpu',
            self.lrank
        )
        if self.cuda:
            torch.cuda.set_device(self.lrank)

        if self.testrun:
            # deterministic testrun
            torch.manual_seed(self.seed)
            self.g = torch.Generator()
            self.g.manual_seed(self.seed)
            if self.cuda:
                torch.cuda.manual_seed(self.seed)

        self.logger = (
            logger if logger is not None
            else BaseLogger(create_new=self.grank == 0)
        )

    @property
    def backend(self) -> str:
        return self._backend

    @backend.setter
    def backend(self, backend_name: str) -> None:
        if backend_name not in TorchDistributedBackend:
            raise ValueError(
                "Unrecognized 'backend' field. Allowed values "
                f"are: {TorchDistributedBackend.list()}")
        self._backend = backend_name

    @property
    def strategy(self) -> Optional[str]:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_name) -> None:
        if strategy_name not in TorchDistributedStrategy:
            raise ValueError(
                "Unrecognized 'strategy' field. Allowed values "
                f"are: {TorchDistributedStrategy.list()}")
        self._strategy = strategy_name

    @property
    def loss(self) -> Callable:
        return self._loss

    @loss.setter
    def loss(self, loss: Loss) -> None:
        if loss is not None:
            self._loss = loss
        else:
            raise ValueError("Loss cannot be None")

    def setup(self, args) -> None:
        pass

    def execute(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    def _train(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None,
    ):
        self.model = self.model.to(self.device)
        # Create distributed model
        if self.strategy == TorchDistributedStrategy.NONE.value:
            pass
        elif self.strategy == TorchDistributedStrategy.DDP.value:
            if not self.cuda:
                raise RuntimeError(
                    "Cannot use torch distributed data parallel without CUDA."
                )
            self.model = DDP(
                self.model,
                device_ids=[self.device],
                output_device=self.device
            )
        else:
            raise NotImplementedError("Only DDP strategy is implemented.")

        # Optimizers
        self.optimizer = self.configure_optimizers(
            optim=optimizer, optim_kwargs=optimizer_kwargs)

        # Dataloaders
        self.train_dataloader = self._preproc_dataloader(train_dataloader)
        if validation_dataloader is not None:
            self.validation_dataloader = self._preproc_dataloader(
                validation_dataloader
            )
        return self.train(model, train_dataloader, validation_dataloader)

    def _preproc_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Make dataloader distributed if using distributed training strategy.

        Args:
            dataloader (DataLoader): some torch DataLoader instance.
        """
        generator = self.g if self.testrun else dataloader.generator

        if self.strategy is None:
            # No distributed strategy used.
            return DataLoader(
                dataloader.dataset,
                shuffle=self.shuffle_data,
                batch_size=dataloader.batch_size,
                sampler=dataloader.sampler,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
                pin_memory=dataloader.pin_memory,
                drop_last=dataloader.drop_last,
                timeout=dataloader.timeout,
                worker_init_fn=seed_worker,  # dataloader.worker_init_fn,
                multiprocessing_context=dataloader.multiprocessing_context,
                generator=generator,
                prefetch_factor=dataloader.prefetch_factor,
                persistent_workers=dataloader.persistent_workers,
                pin_memory_device=dataloader.pin_memory_device
            )
        else:
            sampler = DistributedSampler(
                dataloader.dataset,
                num_replicas=self.gwsize,
                rank=self.grank,
                shuffle=self.shuffle_data
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
            generator=generator,
            prefetch_factor=dataloader.prefetch_factor,
            persistent_workers=dataloader.persistent_workers,
            pin_memory_device=dataloader.pin_memory_device
        )

    def training_step(self, batch, batch_idx) -> Loss:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        pred_y = self.model(x)
        return self.loss(pred_y, y)

    def validation_step(self, batch, batch_idx) -> Loss:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        pred_y = self.model(x)
        return self.loss(pred_y, y)

    def training_epoch(self, epoch_idx) -> Loss:
        self.model.train()
        train_losses = []
        # TODO: use tqdm
        for tr_b_idx, train_batch in enumerate(self.train_dataloader):
            loss = self.training_step(
                batch=train_batch,
                batch_idx=tr_b_idx
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss)
        avg_loss = torch.mean(torch.stack(train_losses)).detach().cpu()
        print(f"Avg train loss: {avg_loss}")
        return avg_loss

    def validation_epoch(self, epoch_idx) -> Loss:
        if self.validation_dataloader is not None:
            self.model.eval()
            validation_losses = []
            # TODO: use tqdm
            for val_b_idx, val_batch in enumerate(self.validation_dataloader):
                loss = self.validation_step(
                    batch=val_batch,
                    batch_idx=val_b_idx
                )
                validation_losses.append(loss)
            avg_loss = torch.mean(
                torch.stack(validation_losses)
            ).detach().cpu()
            print(f"Avg validation loss: {avg_loss}")
            return avg_loss

    def train(self):

        if self.optimizer is None:
            raise ValueError("Undefined optimizer!")

        if self.loss is None:
            raise ValueError("Undefined loss function!")

        st = time.time()

        # Resume state
        start_epoch = 1
        best_loss = np.Inf
        res_name = os.path.join(self.logger.run_path, 'checkpoint.pth.tar')
        if os.path.isfile(res_name) and not self.benchrun:
            try:
                if torch.cuda.is_available():
                    dist.barrier()
                    # Map model to be loaded to specified single gpu.
                    loc = {'cuda:%d' % 0: 'cuda:%d' % self.lrank} if self.cuda else {
                        'cpu:%d' % 0: 'cpu:%d' % self.lrank}
                    checkpoint = torch.load(res_name, map_location=loc)
                else:
                    checkpoint = torch.load(res_name, map_location='cpu')
                start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if torch.cuda.is_available():
                    if self.grank == 0:
                        print(f'WARNING: restarting from {start_epoch} epoch')
                else:
                    print(f'WARNING: restarting from {start_epoch} epoch')
            except:
                if torch.cuda.is_available():
                    if self.grank == 0:
                        print('WARNING: restart file cannot be loaded, restarting!')
                else:
                    print('WARNING: restart file cannot be loaded, restarting!')

        if start_epoch >= self.epochs + 1:
            if torch.cuda.is_available():
                if self.grank == 0:
                    print('WARNING: given epochs are less than the one in the restart file!\n'
                          'WARNING: SYS.EXIT is issued')
                dist.destroy_process_group()
                sys.exit()
            else:
                print('WARNING: given epochs are less than the one in the restart file!\n'
                      'WARNING: SYS.EXIT is issued')
                sys.exit()

        # start trainin/testing loop
        if self.grank == 0:
            print('TIMER: broadcast:', time.time()-st, 's')
            print(f'\nDEBUG: start training')
            print(f'--------------------------------------------------------')

        et = time.time()
        # TODO use tqdm? For distributed situations could be difficult
        for epoch_idx in range(start_epoch, self.epochs + 1):
            lt = time.time()

            if self.benchrun and epoch_idx == self.epochs:
                # profiling (done on last epoch - slower!)
                with torch.autograd.profiler.profile(use_cuda=self.cuda,
                                                     profile_memory=True) as prof:
                    train_loss = self.training_epoch(epoch_idx=epoch_idx)
            else:
                train_loss = self.training_epoch(epoch_idx=epoch_idx)
            val_loss = self.validation_epoch(epoch_idx=epoch_idx)

            # save first epoch timer
            if epoch_idx == start_epoch:
                first_ep_t = time.time()-lt

            # final epoch
            if epoch_idx + 1 == self.epochs:
                self.train_dataloader.last_epoch = True
                self.validation_dataloader.last_epoch = True

            if self.grank == 0:
                print('TIMER: epoch time:', time.time()-lt, 's')
                if self.benchrun and epoch_idx == self.epochs:
                    print('\n--------------------------------------------------------')
                    print('DEBUG: benchmark of last epoch:\n')
                    what1 = 'cuda' if self.cuda else 'cpu'
                    print(prof.key_averages().table(
                        sort_by='self_'+str(what1)+'_time_total'))

            # save state if found a better state
            ref_loss = val_loss if val_loss is not None else train_loss
            is_best = ref_loss < best_loss
            if epoch_idx % self.checkpoint_every == 0 and not self.benchrun:
                save_state(
                    epoch_idx, self.model, ref_loss, self.optimizer,
                    res_name, self.grank, self.gwsize, is_best
                )
                # reset best_acc
                best_loss = min(ref_loss, best_loss)

        # save final state
        if not self.benchrun:
            save_state(
                epoch_idx, self.model, ref_loss,
                self.optimizer, res_name, self.grank, self.gwsize, True
            )
        if torch.cuda.is_available():
            dist.barrier()

        # some debug
        if self.grank == 0:
            print('\n--------------------------------------------------------')
            print('DEBUG: training results:\n')
            print('TIMER: first epoch time:', first_ep_t, ' s')
            print('TIMER: last epoch time:', time.time()-lt, ' s')
            print('TIMER: average epoch time:',
                  (time.time()-et)/self.epochs, ' s')
            print('TIMER: total epoch time:', time.time()-et, ' s')
            if epoch_idx > 1:
                print('TIMER: total epoch-1 time:',
                      time.time()-et-first_ep_t, ' s')
                print('TIMER: average epoch-1 time:',
                      (time.time()-et-first_ep_t)/(self.epochs-1), ' s')
            if self.benchrun:
                print('TIMER: total epoch-2 time:', lt-first_ep_t, ' s')
                print('TIMER: average epoch-2 time:',
                      (lt-first_ep_t)/(self.epochs-2), ' s')
            print('DEBUG: memory req:', int(torch.cuda.memory_reserved(self.lrank)/1024/1024), 'MB') \
                if self.cuda else 'DEBUG: memory req: - MB'
            print('DEBUG: memory summary:\n\n',
                  torch.cuda.memory_summary(0)) if self.cuda else ''

        if self.grank == 0:
            print(f'TIMER: final time: {time.time()-st} s\n')

        # TODO: use a with?
        self.cleanup()

    def cleanup(self):
        """
        Destroy a given process group, and deinitialize the distributed
        package.
        """
        if torch.cuda.is_available():
            dist.barrier()
            dist.destroy_process_group()
