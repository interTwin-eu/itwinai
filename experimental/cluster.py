import abc
import os

from lightning.pytorch.plugins.environments import (
    ClusterEnvironment as LightningClusterEnvironment,
    SLURMEnvironment as LightningSLURMEnvironment,
    TorchElasticEnvironment as LightningTorchElasticEnvironment,
    LightningEnvironment
)


class ClusterEnvironment(LightningClusterEnvironment):
    @abc.abstractmethod
    def num_nodes(self) -> int:
        """Returns the number of nodes allocated for the current job."""


class SLURMEnvironment(LightningSLURMEnvironment):
    def num_nodes(self) -> int:
        """Returns the number of nodes allocated for the current job."""
        if os.environ.get('SLURM_JOB_NUM_NODES'):
            return int(os.environ['SLURM_JOB_NUM_NODES'])
        return int(os.environ['SLURM_NNODES'])


class TorchElasticEnvironment(LightningTorchElasticEnvironment):
    def num_nodes(self) -> int:
        """Returns the number of nodes allocated for the current job."""
        gwsize = int(os.environ['WORLD_SIZE'])
        lwsize = int(os.environ['LOCAL_WORLD_SIZE'])
        return gwsize//lwsize


class LocalEnvironment(LightningEnvironment):
    def num_nodes(self) -> int:
        """Returns the number of nodes allocated for the current job."""
        return 1
