# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Tests for MNIST use case.

Intended to be integration tests, to make sure that updates in the code base
do not break use cases' workflows.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

MNIST_FOLDER = Path("use-cases", "mnist")
TORCH_PATH = MNIST_FOLDER / "torch"
LIGHTNING_PATH = MNIST_FOLDER / "torch-lightning"
TF_PATH = MNIST_FOLDER / "tensorflow"

DEFAULT_MNIST_DATASET = ".tmp"


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

    If MNIST_DATASET env variable is defined, it is used to
    override the default dataset download location: useful
    when it contains a local copy of the dataset, preventing
    downloading it again.
    """
    install_requirements(TORCH_PATH, torch_env)

    dataset_path = os.environ.get("MNIST_DATASET", DEFAULT_MNIST_DATASET)

    conf = (TORCH_PATH / "config.yaml").resolve()
    cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline "
        f"--config {conf} --pipe-key training_pipeline "
        f"-o dataset_root={dataset_path}"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(cmd.split(), check=True, cwd=temp_dir)


@pytest.mark.functional
def test_mnist_inference_torch(torch_env, install_requirements):
    """
    Test MNIST torch native inference by running it end-to-end.
    """
    install_requirements(TORCH_PATH, torch_env)

    conf = (TORCH_PATH / "config.yaml").resolve()
    exec = (TORCH_PATH / "create_inference_sample.py").resolve()

    run_inference_cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline "
        f"--config {conf} --pipe-key inference_pipeline"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create fake inference dataset and checkpoint
        generate_model_cmd = f"{torch_env}/bin/python {exec} " f"--root {temp_dir}"
        subprocess.run(generate_model_cmd.split(), check=True, cwd=temp_dir)

        # Running inference
        subprocess.run(run_inference_cmd.split(), check=True, cwd=temp_dir)


@pytest.mark.functional
def test_mnist_train_torch_lightning(torch_env, install_requirements):
    """
    Test MNIST torch lightning trainer by running it end-to-end.

    If MNIST_DATASET env variable is defined, it is used to
    override the default dataset download location: useful
    when it contains a local copy of the dataset, preventing
    downloading it again.
    """
    install_requirements(LIGHTNING_PATH, torch_env)

    dataset_path = os.environ.get("MNIST_DATASET", DEFAULT_MNIST_DATASET)
    conf = (LIGHTNING_PATH / "config.yaml").resolve()
    cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline "
        f"--config {conf} --pipe-key training_pipeline "
        f"-o dataset_root={dataset_path}"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(cmd.split(), check=True, cwd=temp_dir)


@pytest.mark.functional
def test_mnist_train_tf(tf_env, install_requirements):
    """
    Test MNIST tensorflow trainer by running it end-to-end.
    """
    install_requirements(TF_PATH, tf_env)
    conf = TF_PATH / "pipeline.yaml"
    cmd = f"{tf_env}/bin/itwinai exec-pipeline " f"--config {conf.resolve()} --pipe-key pipeline"
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(cmd.split(), check=True, cwd=temp_dir)
