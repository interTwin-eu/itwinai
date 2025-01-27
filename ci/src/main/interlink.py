import contextlib
import dataclasses
from typing import Annotated

import dagger
import yaml
from dagger import Doc, dag, function, object_type

from .k8s import K8sClient, create_pod_manifest


@object_type
class InterLinkService:
    """Wrapper for interLink service, usefult to communciate with the interLink VK."""

    values: Annotated[dagger.Secret, Doc("interLink installer configuration")]
    name: Annotated[
        str, Doc("Name of the k3s cluster in which the interLink VK is deployed")
    ] = "interlink-cluster"
    wait: Annotated[
        int,
        Doc("Sleep time (seconds) needed to wait for the VK to appear in the k3s cluster."),
    ] = 60

    kubeconfig: Annotated[dagger.File | None, Doc("kubeconfig for k3s cluster")] = (
        dataclasses.field(init=False, default=None)
    )
    _service: dagger.Service | None = dataclasses.field(init=False, default=None)
    vk_name: str | None = dataclasses.field(init=False, default=None)

    @contextlib.asynccontextmanager
    async def start_serving(self) -> dagger.Service:
        """Start and stop interLink service."""
        yield await self.start_service()
        await self.stop_service()

    @function
    async def start_service(self) -> dagger.Service:
        """Create and return an interLink service (k3s service). Can be used in the CLI
        with "<service> up". Example:

        >>> dagger call interlink --values=file:tmp.yaml start-service up
        """
        # Get VK name
        values_dict = yaml.safe_load(await self.values.plaintext())
        self.vk_name = values_dict["nodeName"]

        # Start service
        self._service: dagger.Service = dag.interlink(name=self.name).interlink_cluster(
            values=self.values, wait=self.wait
        )
        self.kubeconfig = dag.interlink(name=self.name).cluster_config(local=False)
        return await self._service.start()

    @function
    async def stop_service(self) -> dagger.Service:
        """Stop the interLink service based on k3s.

        Returns:
            dagger.Service: k3s service.
        """
        return await self._service.stop()

    @function
    async def client(self) -> dagger.Container:
        """Create an interLink service (k3s service) and return the container of the k8s
        client, which can be chanined in the CLI, e.g., with a terminal. Example:

        >>> dagger call interlink --values=file:tmp.yaml client teminal
        """
        await self.start_service()
        k8s_client = K8sClient(kubeconfig=self.kubeconfig)
        return k8s_client.container()

    @function
    async def test_offloading(self) -> str:
        """Test container offloading mechanism on remote HPC using interLink by runnign simple
        tests."""
        # Create pod manifest
        gpus_per_node = 1
        cpus_per_gpu = 1

        # Request memory and CPUs
        resources = {
            "limits": {"cpu": 48, "memory": "150Gi"},
            "requests": {"cpu": cpus_per_gpu * gpus_per_node, "memory": "20Gi"},
        }
        annotations = {
            "slurm-job.vk.io/flags": (
                # --cpus-per-gpu fails on Vega through interLink
                # Use the dev partition for these simple tests
                f"-p dev --gres=gpu:{gpus_per_node} --gpus-per-node={gpus_per_node} "
                "--ntasks-per-node=1 --nodes=1 "
                f"--cpus-per-task={cpus_per_gpu * gpus_per_node} "
                "--time=00:15:00"
            )
        }
        image_path = "/ceph/hpc/data/st2301-itwin-users/cern/hello-world-image.sif"
        pod_name = "interlink-test-offloading"

        # Launch interLink service
        async with self.start_serving():
            k8s_client = K8sClient(kubeconfig=self.kubeconfig)

            stdout = []
            stdout.append(await k8s_client.status())

            # Test #1: test successful pod
            stdout.append("### Test succesful pod ###")
            full_name = pod_name + "-test-succeeded"
            pod_manifest = create_pod_manifest(
                annotations=annotations,
                image_path=image_path,
                cmd_args=["sleep 5 && ls -la"],
                name=full_name,
                resources=resources,
                target_node=self.vk_name,
            )
            pod_manifest_str = yaml.dump(pod_manifest)

            stdout.append(await k8s_client.submit_pod(pod_manifest_str))
            # await k8s_client.container().terminal()
            status, logs = await k8s_client.wait_pod(
                name=full_name, timeout=300, poll_interval=5
            )
            assert (
                status == "Succeeded"
            ), f"Pod did not complete successfully as expected. Got status: {status}"
            stdout.extend([status, logs])

            # Test #2: test pod expected to fail
            stdout.append("### Test failing pod ###")
            full_name = pod_name + "-test-failed"
            pod_manifest = create_pod_manifest(
                annotations=annotations,
                image_path=image_path,
                cmd_args=["sleep 5 && exit 2"],
                name=full_name,
                resources=resources,
                target_node=self.vk_name,
            )
            pod_manifest_str = yaml.dump(pod_manifest)

            stdout.append(await k8s_client.submit_pod(pod_manifest_str))
            # await k8s_client.container().terminal()
            status, logs = await k8s_client.wait_pod(
                name=full_name, timeout=300, poll_interval=5
            )
            assert status == "Failed", f"Pod did not fail as expected. Got status: {status}"
            stdout.extend([status, logs])

            # Test #3: test pod expected to timeout
            stdout.append("### Test timed out pod ###")
            full_name = pod_name + "-test-timeout"
            pod_manifest = create_pod_manifest(
                annotations=annotations,
                image_path=image_path,
                cmd_args=["sleep 10"],
                name=full_name,
                resources=resources,
                target_node=self.vk_name,
            )
            pod_manifest_str = yaml.dump(pod_manifest)

            stdout.append(await k8s_client.submit_pod(pod_manifest_str))
            # await k8s_client.container().terminal()
            status, logs = await k8s_client.wait_pod(
                name=full_name, timeout=3, poll_interval=1
            )
            assert status in (
                "Pod timed out with status: Pending",
                "Pod timed out with status: Running",
            ), f"Pod was expected to run out of time but it didn't. Got status: {status}"
            stdout.extend([status, logs])

        return "\n\n".join(stdout)
