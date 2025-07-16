# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import contextlib
import dataclasses
import time
from datetime import datetime
from typing import Annotated

import dagger
import yaml
from dagger import Doc, dag, function, object_type

from .k8s import create_pod_manifest


@object_type
class InterLink:
    """Wrapper for interLink service, usefult to communciate with the interLink VK."""

    values: Annotated[dagger.Secret, Doc("interLink installer configuration")]
    name: Annotated[
        str, Doc("Name of the k3s cluster in which the interLink VK is deployed")
    ] = "interlink-cluster"
    wait: Annotated[
        int,
        Doc("Sleep time (seconds) needed to wait for the VK to appear in the k3s cluster."),
    ] = 60
    vk_name: Annotated[
        str | None,
        Doc("Name of the interLink VK. Automatically detected from 'values' if not given"),
    ] = None
    kubernetes: Annotated[
        dagger.Service | None,
        Doc(
            "Endpoint to exsisting k3s service to attach to. "
            "Example (from the CLI): tcp://localhost:1997"
        ),
    ] = None

    _service: dagger.Service | None = dataclasses.field(init=False, default=None)

    @function
    async def cluster_config(
        self,
        local: Annotated[bool, Doc("Whether to access the cluster from localhost.")] = False,
    ) -> dagger.File:
        """Returns the config file for the k3s cluster."""
        k3s = dag.k3_s(self.name)

        return k3s.config(local=local)

    @function
    async def interlink_cluster(self) -> dagger.Service:
        """Get interLink VK deployed on a k3s cluster. Returns k3s server as a service.

        This is the first method you want to call from this class to setup the k3s cluster
        and deploy the VK if it does not exist yet. However, if an existing service is
        available, it will reuse it.
        """

        if not self.vk_name:
            values_dict = yaml.safe_load(await self.values.plaintext())
            self.vk_name = values_dict["nodeName"]

        if self.kubernetes:
            self._service = self.kubernetes

        if self._service:
            # Service was already initialized
            return self._service

        k3s = dag.k3_s(self.name)
        server = k3s.server()

        self._service = await server.start()

        # Deploy interLink VK in the k3s cluster
        # TODO: here I could have additional deployments in the cluster, e.g., a jupyterhub
        # deployment and they should be activated or not using simple bool args from this
        # function. Example interlink-cluster --jupyterhub
        await (
            dag.container()
            .from_("alpine/helm")
            .with_exec(["apk", "add", "kubectl"])
            .with_mounted_file("/.kube/config", k3s.config())
            .with_mounted_secret("/values.yaml", self.values)
            .with_env_variable("KUBECONFIG", "/.kube/config")
            .with_exec(
                [
                    "helm",
                    "install",
                    "--debug",
                    "my-node",
                    "oci://ghcr.io/interlink-hq/interlink-helm-chart/interlink",
                    "--values",
                    "/values.yaml",
                ]
            )
            # Wait for the VK to appear as a node
            .with_exec(["sleep", f"{self.wait}"])
            .stdout()
        )

        await k3s.kubectl("wait --for=condition=Ready nodes --all --timeout=300s").stdout()

        return self._service

    @contextlib.asynccontextmanager
    async def start_serving(self) -> dagger.Service:
        """Start and stop interLink service."""
        yield await self.interlink_cluster()
        await self.teardown()

    @function
    async def teardown(self) -> None:
        """Stop the k3s service on which interLink VK is running.

        Returns:
            dagger.Service: k3s service.
        """
        await self._service.stop()
        self._service = None

    @function
    async def test_offloading(
        self,
        partition: Annotated[
            str, Doc("HPC partition on which to test the offloading")
        ] = "dev",
    ) -> str:
        """Test container offloading mechanism on remote HPC using interLink by running simple
        tests."""

        # Request memory and CPUs
        n_cpus = 4
        resources = {
            "limits": {"cpu": 8, "memory": "16Gi"},
            "requests": {"cpu": n_cpus, "memory": "10Gi"},
        }
        annotations = {
            "slurm-job.vk.io/flags": (
                # Use the dev partition for these simple tests
                f"-p {partition} --ntasks-per-node=1 --nodes=1 "
                f"--cpus-per-task={n_cpus} --time=00:15:00"
            )
        }
        if partition == "gpu":
            # Add requests for GPU as well, otherwise the request may hang forever
            annotations["slurm-job.vk.io/flags"] += " --gres=gpu:1 --gpus-per-node=1"
        image_path = "/ceph/hpc/data/st2301-itwin-users/cern/hello-world-image.sif"
        pod_name = "interlink-test-offloading"

        # Launch interLink service
        async with self.start_serving():
            assert self.vk_name is not None, "vk_name is none!"

            stdout = []
            stdout.append(await self.status())

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

            stdout.append(await self.submit_pod(pod_manifest_str))
            # await k8s_client.container().terminal()
            status, logs = await self.wait_pod(name=full_name, timeout=300, poll_interval=5)
            assert status == "Succeeded", (
                f"Pod did not complete successfully as expected. Got status: {status}"
            )
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

            stdout.append(await self.submit_pod(pod_manifest_str))
            # await k8s_client.container().terminal()
            status, logs = await self.wait_pod(name=full_name, timeout=300, poll_interval=5)
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

            stdout.append(await self.submit_pod(pod_manifest_str))
            # await k8s_client.container().terminal()
            status, logs = await self.wait_pod(name=full_name, timeout=3, poll_interval=1)
            assert status in (
                "Pod timed out with status: Pending",
                "Pod timed out with status: Running",
            ), f"Pod was expected to run out of time but it didn't. Got status: {status}"
            stdout.extend([status, logs])

        return "\n\n".join(stdout)

    @function
    async def client(self) -> dagger.Container:
        """Returns a client for the k3s cluster. If the cluster does not exist,
        it will create it.

        >>> dagger call interlink --values=file:tmp.yaml client teminal
        """
        await self.interlink_cluster()
        return dag.k3_s(name=self.name).container()

    async def _run_cmd(self, cmd: list[str], stdin: str | None = None) -> str:
        container = await self.client()
        return await (
            container
            # Invalidate cache
            .with_env_variable("CACHE", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
            .with_exec(cmd, stdin=stdin)
            .stdout()
        )

    @function
    async def status(self) -> str:
        """Get k8s cluster status, including nodes and pods under default namespace."""
        pods = await self._run_cmd("kubectl get pods".split())
        nodes = await self._run_cmd("kubectl get nodes".split())
        return f"Nodes:\n{nodes}\n\nPods:\n{pods}"

    @function
    async def submit_pod(
        self, manifest: Annotated[str, Doc("pod manifest serialized as string.")]
    ) -> str:
        """Submit pod to k8s cluster. Returns the result of submission."""
        cmd = ["kubectl", "apply", "-f", "-"]
        return await self._run_cmd(cmd, stdin=manifest)

    @function
    async def get_pod_status(self, name: Annotated[str, Doc("pod name")]) -> str:
        """Get pod status: Pending, Running, Succeeded, Failed, Unknown.
        Returns the pod status in that moment.
        """
        cmd = [
            "kubectl",
            "get",
            "pod",
            f"{name}",
            "-n",
            "default",
            "-o",
            "jsonpath='{.status.phase}'",
        ]
        status = await self._run_cmd(cmd)
        return status.strip("'")

    @function
    async def get_pod_logs(self, name: Annotated[str, Doc("pod name")]) -> str:
        """Get pod logs."""
        cmd = [
            "kubectl",
            "logs",
            "--insecure-skip-tls-verify-backend",
            f"{name}",
            "-n",
            "default",
        ]
        return await self._run_cmd(cmd)

    @function
    async def delete_pod(self, name: Annotated[str, Doc("pod name")]) -> str:
        """Delete pod. Returns the result of ``kubectl delete pod``."""
        cmd = [
            "kubectl",
            "delete",
            "pod",
            f"{name}",
            "-n",
            "default",
            "--grace-period=0",
            "--force",
        ]
        return await self._run_cmd(cmd)

    @function
    async def wait_pod(
        self,
        name: Annotated[str, Doc("pod name")],
        timeout: Annotated[int, Doc("timeout in seconds")] = 300,
        poll_interval: Annotated[int, Doc("how often to check the pod status")] = 5,
    ) -> list[str]:
        """Wait for pod termination (Succeeded, Failed) or timeout.
        Returns last pod status and logs detected.
        """
        counter = 0
        timeout = max(int(timeout / poll_interval), 1)
        # Allow at most about 60 seconds of unk status
        unk_timeout = max(int(60 / poll_interval), 1)
        try:
            while True:
                status = await self.get_pod_status(name=name)
                logs = await self.get_pod_logs(name=name)
                if status in ["Succeeded", "Failed"]:
                    break
                counter += 1
                if counter > timeout or status == "Unknown" and counter > unk_timeout:
                    status = f"Pod timed out with status: {status}"
                    break
                time.sleep(poll_interval)
        finally:
            await self.delete_pod(name=name)
        return [status, logs]
