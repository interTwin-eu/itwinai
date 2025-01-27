# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import copy
import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import dagger
import yaml
from dagger import dag
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Template with default values
POD_TEMPLATE = {
    "apiVersion": "v1",
    "kind": "Pod",
    "metadata": {
        "name": "pod-template",
        "namespace": "default",
        "annotations": {
            "slurm-job.vk.io/flags": (
                "-p gpu --gres=gpu:1 --ntasks-per-node=1 --nodes=1 --time=00:10:00"
            ),
            "slurm-job.vk.io/pre-exec": "ls -la",
        },
    },
    "spec": {
        "containers": [
            {
                "name": "busy-container",
                "image": "/ceph/hpc/data/st2301-itwin-users/cern/hello-world-image.sif",
                "command": ["/bin/sh", "-c"],
                "args": ["sleep $(( 60 * 1 ))"],
                "imagePullPolicy": "Always",
                "resources": {
                    "limits": {"cpu": "48", "memory": "150Gi"},
                    "requests": {"cpu": "4", "memory": "20Gi"},
                },
            }
        ],
        "restartPolicy": "Always",
        "nodeSelector": {"kubernetes.io/hostname": "vega-new-vk"},
        "tolerations": [
            {"key": "virtual-node.interlink/no-schedule", "operator": "Exists"},
            {
                "effect": "NoExecute",
                "key": "node.kubernetes.io/not-ready",
                "operator": "Exists",
                "tolerationSeconds": 300,
            },
            {
                "effect": "NoExecute",
                "key": "node.kubernetes.io/unreachable",
                "operator": "Exists",
                "tolerationSeconds": 300,
            },
        ],
    },
}


def validate_pod_name(name: str) -> None:
    """Validates if a given string matches the Kubernetes pod naming convention.

    Args:
        name (str): The pod name to validate.

    Raises:
        ValueError: If the name does not match the required pattern.
    """
    pattern = r"[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*"
    if not re.fullmatch(pattern, name):
        raise ValueError(
            f"The name '{name}' is invalid for a Kubernetes pod. "
            "Pod names must match the pattern: "
            "'[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*'"
        )


def create_pod_manifest(
    name: str = None,
    annotations: Dict = None,
    container_name: str = None,
    image_path: str = None,
    cmd_args: List[str] = None,
    resources: Dict = None,
    target_node: str = None,
):
    """Creates a pod manifest with optional parameters to override default values
    for pod configuration.

    Args:
        name (str, optional): Name of the pod (metadata.name).
        annotations (Dict, optional): Annotations for the pod (metadata.annotations).
        container_name (str, optional): Name of the container (spec.containers[0].name).
        image_path (str, optional): Path to the container image
            (spec.containers[0].image).
        cmd_args (List[str], optional): Command arguments for the container
            (spec.containers[0].args).
        resources (Dict, optional): Resource requests and limits for the container
            (spec.containers[0].resources).
        target_node (str): target node of the node selector.

    Returns:
        Dict: A dictionary representing the pod manifest.
    """

    # Deep copy the template to avoid modifying the original
    pod_manifest = copy.deepcopy(POD_TEMPLATE)

    # Set metadata fields
    if name:
        validate_pod_name(name)
        pod_manifest["metadata"]["name"] = name
    if annotations:
        pod_manifest["metadata"]["annotations"].update(annotations)

    # Set container-specific fields
    container = pod_manifest["spec"]["containers"][0]
    if container_name:
        container["name"] = container_name
    if image_path:
        container["image"] = image_path
    if cmd_args:
        container["args"] = cmd_args
    if resources:
        container["resources"] = resources

    if target_node:
        pod_manifest["spec"]["nodeSelector"]["kubernetes.io/hostname"] = target_node

    return pod_manifest


def load_kube_config_from_string(kubeconfig_str: str):
    kubeconfig_dict = yaml.safe_load(kubeconfig_str)
    config.load_kube_config_from_dict(kubeconfig_dict)


def load_pod_manifest_from_yaml(file_path: str) -> Dict:
    """Load a pod manifest from a YAML file."""
    with open(file_path, "r") as f:
        pod_manifest = yaml.safe_load(f)
    return pod_manifest


def create_pod(api_instance: client.CoreV1Api, namespace: str, pod_manifest: Dict):
    try:
        api_response = api_instance.create_namespaced_pod(
            namespace=namespace, body=pod_manifest
        )
        print(f"Pod created. Status: {api_response.status.phase}")
    except ApiException as e:
        print(f"Exception when creating pod: {e}")


def check_pod_status(api_instance: client.CoreV1Api, namespace: str, pod_name: str):
    try:
        pod = api_instance.read_namespaced_pod(name=pod_name, namespace=namespace)
        return pod.status.phase
    except ApiException as e:
        print(f"Exception when checking pod status: {e}")
        return None


def get_pod_logs_insecure(api_instance: client.CoreV1Api, namespace: str, pod_name: str):
    """Fetch logs for the specified pod with insecure TLS settings."""
    try:
        log_response = api_instance.read_namespaced_pod_log(
            name=pod_name, namespace=namespace, insecure_skip_tls_verify_backend=True
        )
        return log_response
    except ApiException as e:
        print(f"Exception when retrieving pod logs: {e}")
        return None


def delete_pod(api_instance: client.CoreV1Api, namespace: str, pod_name: str):
    """Delete a pod by its name in a specified namespace."""
    try:
        api_response = api_instance.delete_namespaced_pod(name=pod_name, namespace=namespace)
        print(f"Pod '{pod_name}' deleted. Status: {api_response.status}")
    except ApiException as e:
        print(f"Exception when deleting pod: {e}")


def submit_job(
    kubeconfig_str: str, pod_manifest: Dict, verbose: bool = False
) -> Optional[str]:
    """Submit a pod to some kubernetes cluster.

    Args:
        kubeconfig_str (str): kubeconfig for kubernetes in string format.
        pod_manifest (Dict): manifest of the pod.
        verbose (bool, optional): whether to print pod logs every second. Defaults to False.

    Returns:
        Optional[str]: final pod status.
    """
    load_kube_config_from_string(kubeconfig_str)

    # Initialize the CoreV1Api for managing pods
    v1 = client.CoreV1Api()

    namespace = "default"
    pod_name = pod_manifest["metadata"]["name"]

    # Kill existing pod, if present
    status = check_pod_status(v1, namespace, pod_name)
    if status:
        logging.warning(f"Pod {pod_name} already existed... Deleting it before continuing.")
        delete_pod(v1, namespace, pod_name)
        while status is not None:
            time.sleep(1)
            status = check_pod_status(v1, namespace, pod_name)

    # Create the pod in the specified namespace
    create_pod(v1, namespace, pod_manifest)
    status = None
    logs = None
    try:
        # Check the pod status
        status = check_pod_status(v1, namespace, pod_name)
        print(f"Pod status: {status}")
        print("=" * 100)
        print("Waiting for pod completion...")

        # Get pod logs with insecure TLS settings if the pod has completed its
        # initialization
        while status in ["Running", "Pending"]:
            time.sleep(1)
            status = check_pod_status(v1, namespace, pod_name)
            logs = get_pod_logs_insecure(v1, namespace, pod_name)
            if verbose:
                print(f"Pod status: {status}")
                print(f"Pod logs:\n{logs}")
    finally:
        print("=" * 100)
        print(f"Pod status: {status}")
        print(f"Pod logs:\n{logs}")
        delete_pod(v1, namespace, pod_name)
    return status


def list_pods(kubeconfig_path: str, namespace: str = "default") -> str:
    """
    Connects to a Kubernetes cluster using the provided kubeconfig file and lists the pods.

    Args:
        kubeconfig_path (str): Path to the kubeconfig file.
        namespace (str): The namespace to list the pods from. Default is 'default'.

    Returns:
        None: Prints the pods in the namespace.
    """
    # Load the kubeconfig file
    config.load_kube_config(config_file=kubeconfig_path)

    # Create a Kubernetes API client
    v1 = client.CoreV1Api()

    report = []

    try:
        # List all the pods in the specified namespace
        report.append(f"Listing pods in namespace '{namespace}':")
        pods = v1.list_namespaced_pod(namespace)
        for pod in pods.items:
            report.append(f"- {pod.metadata.name}")
    except Exception as e:
        report.append(f"Error occurred: {e}")
    return "\n".join(report)


class K8sClient:
    """Kubernetes client providing kubectl to interact with a k8s cluster described
    by some configuration.

    Args:
        kubeconfig (dagger.File): kubeconfig of target cluster.
    """

    kubeconfig: dagger.File
    _k8s_client: dagger.Container = None

    def __init__(self, kubeconfig: dagger.File):
        self.kubeconfig = kubeconfig

    def container(self) -> dagger.Container:
        """Get or create a new k8s client container.

        Returns:
            dagger.Container: container of k8s client.
        """
        if not self._k8s_client:
            self._k8s_client = (
                dag.container()
                .from_("alpine/helm")
                .with_exec(["apk", "add", "kubectl"])
                .with_mounted_file("/.kube/config", self.kubeconfig)
                .with_env_variable("KUBECONFIG", "/.kube/config")
            )

        return self._k8s_client

    async def status(self) -> str:
        """Get k8s cluster status, including nodes and pods under default namespace.

        Returns:
            str: cluster status summary.
        """
        pods = await self.container().with_exec("kubectl get pods".split()).stdout()
        nodes = await self.container().with_exec("kubectl get nodes".split()).stdout()
        return f"Nodes:\n{nodes}\n\nPods:\n{pods}"

    async def submit_pod(self, manifest: str) -> str:
        """Submit pod to k8s cluster

        Args:
            manifest (str): pod manifest serialized as string.

        Returns:
            str: the result of sumission.
        """
        cmd = ["kubectl", "apply", "-f", "-"]
        return await self.container().with_exec(cmd, stdin=manifest).stdout()

    async def get_pod_status(self, name: str) -> str:
        """Get pod status: Pending, Running, Succeeded, Failed, Unknown.

        Args:
            name (str): pod name.

        Returns:
            str: pod status in that moment.
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
        status = await (
            self.container()
            # Invalidate cache
            .with_env_variable("CACHE", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
            .with_exec(cmd)
            .stdout()
        )
        return status.strip("'")

    async def get_pod_logs(self, name: str) -> str:
        """Get pod logs.

        Args:
            name (str): pod name.

        Returns:
            str: pod logs in that moment.
        """
        cmd = [
            "kubectl",
            "logs",
            "--insecure-skip-tls-verify-backend",
            f"{name}",
            "-n",
            "default",
        ]
        logs = await (
            self.container()
            # Invalidate cache
            .with_env_variable("CACHE", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
            .with_exec(cmd)
            .stdout()
        )
        return logs

    async def delete_pod(self, name: str) -> str:
        """Delete pod.

        Args:
            name (str): pod name.

        Returns:
            str: result message of ``kubectl delete pod``.
        """
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
        msg = await (
            self.container()
            # Invalidate cache
            .with_env_variable("CACHE", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
            .with_exec(cmd)
            .stdout()
        )
        return msg

    async def wait_pod(
        self, name: str, timeout: int = 300, poll_interval: int = 5
    ) -> Tuple[str, str]:
        """Wait for pod termination (Succeeded, Failed) or timeout.

        Args:
            name (str): pod name.
            timeout (int, optional): timeout in seconds. Defaults to 300.
            poll_interval (int, optional): how often to check the pod status.
                Defaults to 5 seconds.

        Returns:
            Tuple[str, str]: last pod status and logs detected.
        """
        cnt = 0
        timeout = max(int(timeout / poll_interval), 1)
        # Allow at most about 60 seconds of unk status
        unk_timeout = max(int(60 / poll_interval), 1)
        while True:
            status = await self.get_pod_status(name=name)
            logs = await self.get_pod_logs(name=name)
            if status in ["Succeeded", "Failed"]:
                await self.delete_pod(name=name)
                return status, logs
            cnt += 1
            if cnt > timeout or status == "Unknown" and cnt > unk_timeout:
                await self.delete_pod(name=name)
                return f"Pod timed out with status: {status}", logs
            time.sleep(poll_interval)
