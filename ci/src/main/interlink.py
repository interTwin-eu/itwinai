import contextlib
from typing import Dict

import dagger
from dagger import dag
import yaml


class InterLinkService:
    """Wrapper for interLink service, usefult to communciate with the interLink VK.

    Args:
        values (dagger.File): configuration file for interLink installer.
        name (str, optional): name of the k8s cluster. Defaults to "interlink-svc".
        wait (int, optional): seconds to sleep while waiting for the VK node to become
            ready. Defaults to 60.
    """

    _service: dagger.Service = None
    values: dagger.File
    name: str
    kubeconfig: dagger.File
    vk_name: str

    def __init__(
        self, values: dagger.File, name: str = "interlink-svc", wait: int = 60
    ) -> None:
        self.values = values
        self.name = name
        self.wait = wait

    @contextlib.asynccontextmanager
    async def start_serving(self) -> dagger.Service:
        """Start and stop interLink service."""
        yield await self.start()
        await self.stop()

    async def start(self) -> dagger.Service:
        """Start a new interLink service based on k3s.

        Returns:
            dagger.Service: k3s service.
        """
        # Get VK name
        values_dict = yaml.safe_load(await self.values.contents())
        self.vk_name = values_dict["nodeName"]

        # Start service
        self._service: dagger.Service = dag.interlink(name=self.name).interlink_cluster(
            values=self.values, wait=self.wait
        )
        self.kubeconfig = dag.interlink(name=self.name).cluster_config(local=False)
        return await self._service.start()

    async def stop(self) -> dagger.Service:
        """Stop the interLink service based on k3s.

        Returns:
            dagger.Service: k3s service.
        """
        return await self._service.stop()
