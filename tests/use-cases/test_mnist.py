"""
Tests for MNIST use case.

Intended to be integration tests, to make sure that updates in the code base
do not break use cases' workflows.
"""

import pytest
import subprocess
# from itwinai.cli import exec_pipeline

TORCH_PATH = "use-cases/mnist/torch"
LIGHTNING_PATH = "use-cases/mnist/torch-lightning"
TF_PATH = "use-cases/mnist/tensorflow"


@pytest.mark.skip(reason="structure changed")
def test_structure_mnist_torch(check_folder_structure):
    """Test MNIST folder structure for torch native trainer."""
    check_folder_structure(TORCH_PATH)


@pytest.mark.skip(reason="structure changed")
def test_structure_mnist_lightning(check_folder_structure):
    """Test MNIST folder structure for torch lightning trainer."""
    check_folder_structure(LIGHTNING_PATH)


@pytest.mark.skip(reason="structure changed")
def test_structure_mnist_tf(check_folder_structure):
    """Test MNIST folder structure for tensorflow trainer."""
    check_folder_structure(TF_PATH)


@pytest.mark.functional
def test_mnist_train_torch(torch_env, install_requirements):
    """
    Test MNIST torch native trainer by running it end-to-end.

    To set the torch env path set the ``TORCH_ENV`` env variable:

    >>> export TORCH_ENV="my_env"
    """
    install_requirements(TORCH_PATH, torch_env)
    cmd = (f"{torch_env}/bin/itwinai exec-pipeline "
           f"--config config.yaml --pipe-key training_pipeline")
    subprocess.run(cmd.split(), check=True, cwd=TORCH_PATH)


@pytest.mark.functional
def test_mnist_inference_torch(torch_env, install_requirements):
    """
    Test MNIST torch native inference by running it end-to-end.

    To set the torch env path set the ``TORCH_ENV`` env variable:

    >>> export TORCH_ENV="my_env"
    """
    install_requirements(TORCH_PATH, torch_env)

    # Create fake inference dataset and checkpoint
    cmd = f"{torch_env}/bin/python create_inference_sample.py"
    subprocess.run(cmd.split(), check=True, cwd=TORCH_PATH)

    # Test inference
    cmd = (f"{torch_env}/bin/itwinai exec-pipeline "
           f"--config config.yaml --pipe-key inference_pipeline")
    subprocess.run(cmd.split(), check=True, cwd=TORCH_PATH)


@pytest.mark.functional
def test_mnist_train_torch_lightning(torch_env, install_requirements):
    """
    Test MNIST torch lightning trainer by running it end-to-end.

    To set the torch env path set the ``TORCH_ENV`` env variable:

    >>> export TORCH_ENV="my_env"
    """
    install_requirements(TORCH_PATH, torch_env)
    cmd = (f"{torch_env}/bin/itwinai exec-pipeline "
           f"--config config.yaml --pipe-key training_pipeline")
    subprocess.run(cmd.split(), check=True, cwd=LIGHTNING_PATH)


@pytest.mark.functional
def test_mnist_train_tf(tf_env, install_requirements):
    """
    Test MNIST tensorflow trainer by running it end-to-end.
    """
    install_requirements(TF_PATH, tf_env)
    cmd = (f"{tf_env}/bin/itwinai exec-pipeline "
           f"--config pipeline.yaml --pipe-key pipeline")
    subprocess.run(cmd.split(), check=True, cwd=TF_PATH)
