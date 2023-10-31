"""
Factories to instantiate Launcher classes.
They introduce a level of indirection to provide a unified interface
for all the launchers. The common interface is provided by the
`createLauncher` factory method.
"""

from typing import Optional, Dict, Any
import abc

from launcher import (
    Launcher,
    TorchElasticLauncher,
    SimpleLauncher,
    DeepSpeedLauncher
)
from cluster import detect_cluster


class LauncherFactory(abc.ABC):
    """
    Factory class to instantiate a Launcher classes.
    It introduces a level of indirection to provide a unified interface
    for all the launchers. The common interface is provided by the
    `createLauncher` factory method.
    """

    def createLauncher(
        self,
        n_workers_per_node: int,
        run_id: Optional[str] = None,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        **kwargs
    ) -> Launcher:
        """
        Simplifies the instantiation of a Launcher.
        Advanced configuration is pre-computed in the body
        of this method, leaving few parameters to the end user.
        """


class TorchElasticLauncherFactory(LauncherFactory):
    """Factory class to instantiate a TorchElasticLauncher class."""

    def createLauncher(
        self,
        n_workers_per_node: int,
        run_id: Optional[str] = None,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        **kwargs
    ) -> Launcher:
        """
        Simplifies the instantiation of a TorchElasticLauncher.
        Advanced configuration is pre-computed in the body
        of this method, leaving few parameters to the end user.
        """
        cluster = detect_cluster()

        kwargs['nproc_per_node'] = n_workers_per_node
        # If given, propagate the args
        if run_id:
            kwargs['rdzv_id'] = run_id
        if master_addr:
            kwargs['master_addr'] = master_addr
        if master_port:
            kwargs['master_port'] = master_port

        # Compute and add TorchElastic specific args, if not
        # provided as **kwargs
        n_nodes = cluster.num_nodes()
        safe_add(kwargs, 'nnodes', f"{n_nodes}:{n_nodes}")
        safe_add(kwargs, 'rdzv_id', cluster.job_id())
        is_host_flag = '1' if cluster.node_rank() == 0 else '0'
        safe_add(kwargs, 'rdzv_conf', f'is_host={is_host_flag}')
        safe_add(kwargs, 'rdzv_backend', 'c10d')
        safe_add(
            kwargs,
            'rdzv_endpoint',
            f'{cluster.main_address}:{cluster.main_port}'
        )
        safe_add(kwargs, 'max_restarts', 3)

        return TorchElasticLauncher(**kwargs)


class SimpleLauncherFactory(LauncherFactory):
    """Factory class to instantiate a SimpleLauncherFactory class."""

    def createLauncher(
        self,
        n_workers_per_node: int,
        run_id: Optional[str] = None,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        **kwargs
    ) -> Launcher:
        """
        Simplifies the instantiation of a SimpleLauncher.
        Advanced configuration is pre-computed in the body
        of this method, leaving few parameters to the end user.
        """

        kwargs['nproc_per_node'] = n_workers_per_node
        # If given, propagate the args
        if run_id:
            kwargs['run_id'] = run_id
        if master_addr:
            kwargs['master_addr'] = master_addr
        if master_port:
            kwargs['master_port'] = master_port

        return SimpleLauncher(**kwargs)


class DeepSpeedLauncherFactory(LauncherFactory):
    """Factory class to instantiate a DeepSpeedLauncher class."""

    def createLauncher(
        self,
        n_workers_per_node: int,
        run_id: Optional[str] = None,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        **kwargs
    ) -> Launcher:
        """
        Simplifies the instantiation of a DeepSpeedLauncher.
        Advanced configuration is pre-computed in the body
        of this method, leaving few parameters to the end user.
        """
        # TODO: complete
        raise NotImplementedError
        return DeepSpeedLauncher(...)


def safe_add(map: Dict, key: str, value: Any) -> None:
    """
    Add a key-value pair to a dict if the key
    is not already present.
    """
    if map.get(key) is None:
        map[key] = value
