# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import copy
import re
from typing import Dict, List

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
    name: str | None = None,
    annotations: Dict | None = None,
    container_name: str | None = None,
    image_path: str | None = None,
    cmd_args: List[str] | None = None,
    resources: Dict | None = None,
    target_node: str | None = None,
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
