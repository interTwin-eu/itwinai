import abc
import os
import time

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

    @abc.abstractmethod
    def job_id(self) -> str:
        """Returns the current job ID inferred from the cluster."""


class SLURMEnvironment(LightningSLURMEnvironment):
    def num_nodes(self) -> int:
        """Returns the number of nodes allocated for the current job."""
        if os.environ.get('SLURM_JOB_NUM_NODES'):
            return int(os.environ['SLURM_JOB_NUM_NODES'])
        if os.environ.get('SLURM_NNODES'):
            return int(os.environ['SLURM_NNODES'])
        raise RuntimeError('Number of nodes not found in SLURM env variables')

    def job_id(self) -> str:
        """Returns the current job ID inferred from the cluster."""
        return os.environ['SLURM_JOB_ID']


class TorchElasticEnvironment(LightningTorchElasticEnvironment):
    def num_nodes(self) -> int:
        """Returns the number of nodes allocated for the current job."""
        gwsize = int(os.environ['WORLD_SIZE'])
        lwsize = int(os.environ['LOCAL_WORLD_SIZE'])
        return gwsize//lwsize

    def job_id(self) -> str:
        """Returns the current job ID inferred from the cluster."""
        return os.environ['TORCHELASTIC_RUN_ID']


class LocalEnvironment(LightningEnvironment):

    _job_id: str = None

    def world_size(self) -> int:
        # if os.environ.get('WORLD_SIZE'):
        #     return int(os.environ.get('WORLD_SIZE'))
        print(
            "WARNING: world_size() method in 'LocalEnvironment' returns "
            f"a fixed-value placeholder world_size={self._world_size}. "
            "Use it carefully!"
        )
        return self._world_size

    def global_rank(self) -> int:
        # if os.environ.get('RANK'):
        #     return int(os.environ.get('RANK'))
        print(
            "WARNING: global_rank() method in 'LocalEnvironment' returns "
            f"a fixed-value placeholder global_rank={self._global_rank}. "
            "Use it carefully!"
        )
        return self._global_rank

    def num_nodes(self) -> int:
        """Returns the number of nodes allocated for the current job."""
        return 1

    def job_id(self) -> str:
        """Returns the current job ID inferred from the cluster."""
        if self._job_id is None:
            self._job_id = str(time.time())
        return self._job_id


def detect_cluster() -> ClusterEnvironment:
    """Defines a protocol to select the ClusterEnvironment
    depending on availability and priority.
    """

    if SLURMEnvironment.detect():
        cluster = SLURMEnvironment()
    elif TorchElasticEnvironment.detect():
        cluster = TorchElasticEnvironment()
    elif LocalEnvironment.detect():
        cluster = LocalEnvironment()
    else:
        raise NotImplementedError("Unrecognized cluster env")
    return cluster
