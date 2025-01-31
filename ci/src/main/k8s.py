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
from typing import Dict, List, Optional

import yaml
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

    Returns:
        Dict: A dictionary representing the pod manifest.
    """

    # Deep copy the template to avoid modifying the original
    pod_manifest = copy.deepcopy(POD_TEMPLATE)

    # Set metadata fields
    if name:
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
