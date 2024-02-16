"""
Tests for MNIST use case.

Intended to be integration tests, to make sure that updates in the code base
do not break use cases' workflows.
"""

import pytest
import subprocess

TORCH_PATH = "use-cases/mnist/torch"
LIGHTNING_PATH = "use-cases/mnist/torch-lightning"
TF_PATH = "use-cases/mnist/tensorflow"


def test_structure_mnist_torch(check_folder_structure):
    """Test MNIST folder structure for torch native trainer."""
    check_folder_structure(TORCH_PATH)


def test_structure_mnist_lightning(check_folder_structure):
    """Test MNIST folder structure for torch lightning trainer."""
    check_folder_structure(LIGHTNING_PATH)


def test_structure_mnist_tf(check_folder_structure):
    """Test MNIST folder structure for tensorflow trainer."""
    check_folder_structure(TF_PATH)


@pytest.mark.functional
def test_mnist_train_torch(install_requirements):
    """
    Test MNIST torch native trainer by running it end-to-end.
    """
    install_requirements(TORCH_PATH, pytest.TORCH_PREFIX)
    cmd = (f"micromamba run -p {pytest.TORCH_PREFIX} python "
           f"{TORCH_PATH}/train.py -p {TORCH_PATH}/pipeline.yaml")
    subprocess.run(cmd.split(), check=True)


@pytest.mark.functional
def test_mnist_train_lightning(install_requirements):
    """
    Test MNIST torch lightning trainer by running it end-to-end.
    """
    install_requirements(TORCH_PATH, pytest.TORCH_PREFIX)
    cmd = (f"micromamba run -p {pytest.TORCH_PREFIX} python "
           f"{LIGHTNING_PATH}/train.py -p {LIGHTNING_PATH}/pipeline.yaml")
    subprocess.run(cmd.split(), check=True)


@pytest.mark.functional
def test_mnist_train_tf(install_requirements):
    """
    Test MNIST tensorflow trainer by running it end-to-end.
    """
    install_requirements(TF_PATH, pytest.TF_PREFIX)
    cmd = (f"micromamba run -p {pytest.TF_PREFIX} python "
           f"{TF_PATH}/train.py -p {TF_PATH}/pipeline.yaml")
    subprocess.run(cmd.split(), check=True)


@pytest.mark.skip(reason="workflow changed. Left as example")
@pytest.mark.integration
def test_mnist_train_legacy():
    """
    Test MNIST training workflow(s) by running it end-to-end.
    """
    workflows = [
        "./use-cases/mnist/torch/workflows/training-workflow.yml",
        "./use-cases/mnist/tensorflow/workflows/training-workflow.yml",
    ]

    for workflow in workflows:
        cmd = f"micromamba run -p ./.venv python run-workflow.py -f {workflow}"
        subprocess.run(cmd.split(), check=True)
        subprocess.run(cmd.split() + ["--cwl"], check=True)
