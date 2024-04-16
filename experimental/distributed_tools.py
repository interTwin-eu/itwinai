from typing import Tuple
import abc

from launcher import Launcher
from strategy import Strategy, DDPStrategy
from launcher_factory import TorchElasticLauncherFactory


class Assembler(abc.ABC):
    """Abstract Assembler class."""


class DistributedTooling(Assembler):
    """
    Assembles a set of objects used to enable distributed ML.
    Suggests working presets of Launcher and Strategy, providing
    an easy entry point for the end user.
    """

    def __init__(self, n_workers_per_node: int = 1) -> None:
        super().__init__()
        self.n_workers_per_node = n_workers_per_node

    def getTools(self, strategy: str) -> Tuple[Launcher, Strategy]:
        if strategy == 'ddp':
            return self.getTorchDDPTools()
        if strategy == 'deepspeed':
            return self.getDeepSpeedTools()
        if strategy == 'horovod':
            return self.getHorovodTools()
        raise ValueError(f"Unrecognized strategy={strategy}")

    def getTorchDDPTools(self) -> Tuple[Launcher, Strategy]:
        """
        Returns a suggested preset of Launcher + Strategy
        for torch distributed data parallel.
        """
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Torch DDP cannot be used. GPUs not available."
            )
        if not torch.cuda.device_count() > 1:
            raise RuntimeError(
                "Torch DDP cannot be used. Only one GPU is available."
            )
        launcher_builder = TorchElasticLauncherFactory()
        elastic_launcher = launcher_builder.createLauncher(
            n_workers_per_node=self.n_workers_per_node
        )
        strategy = DDPStrategy(backend='nccl')
        return elastic_launcher, strategy

    def getDeepSpeedTools(self) -> Tuple[Launcher, Strategy]:
        """
        Returns a suggested preset of Launcher + Strategy
        for DeepSpeed distributed ML.
        """
        # TODO: complete
        raise NotImplementedError

    def getHorovodTools(self) -> Tuple[Launcher, Strategy]:
        """
        Returns a suggested preset of Launcher + Strategy
        for Horovod distributed ML.
        """
        # TODO: complete
        raise NotImplementedError
