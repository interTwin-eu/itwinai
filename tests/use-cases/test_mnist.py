# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Tests for MNIST use case.

Intended to be integration tests, to make sure that updates in the code base
do not break use cases' workflows.
"""

import os
import subprocess
from pathlib import Path

import pytest

MNIST_FOLDER = Path("use-cases", "mnist").resolve()
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
def test_mnist_train_torch(torch_env, install_requirements, tmp_path):
    """
    Test MNIST torch native trainer by running it end-to-end.

    If MNIST_DATASET env variable is defined, it is used to
    override the default dataset download location: useful
    when it contains a local copy of the dataset, preventing
    downloading it again.
    """
    install_requirements(TORCH_PATH, torch_env)

    dataset_path = os.environ.get("MNIST_DATASET", DEFAULT_MNIST_DATASET)

    cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline "
        f"--config-path {TORCH_PATH} "
        f"--config-name .config-test.yaml "
        f"dataset_root={dataset_path} "
    )
    subprocess.run(cmd.split(), check=True, cwd=tmp_path)


@pytest.mark.functional
def test_mnist_inference_torch(torch_env, install_requirements, tmp_path):
    """
    Test MNIST torch native inference by running it end-to-end.
    """
    install_requirements(TORCH_PATH, torch_env)

    exec = TORCH_PATH / "create_inference_sample.py"

    run_inference_cmd = f"{torch_env}/bin/itwinai exec-pipeline --config-path {TORCH_PATH} \
        +pipe_key=inference_pipeline"
    # Create fake inference dataset and checkpoint
    generate_model_cmd = f"{torch_env}/bin/python {exec} --root {tmp_path}"
    subprocess.run(generate_model_cmd.split(), check=True, cwd=tmp_path)

    # Running inference
    subprocess.run(run_inference_cmd.split(), check=True, cwd=tmp_path)


@pytest.mark.functional
def test_mnist_train_torch_lightning(torch_env, install_requirements, tmp_path):
    """
    Test MNIST torch lightning trainer by running it end-to-end.

    If MNIST_DATASET env variable is defined, it is used to
    override the default dataset download location: useful
    when it contains a local copy of the dataset, preventing
    downloading it again.
    """
    install_requirements(LIGHTNING_PATH, torch_env)

    dataset_path = os.environ.get("MNIST_DATASET", DEFAULT_MNIST_DATASET)
    cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline "
        f"--config-path {LIGHTNING_PATH} "
        f"dataset_root={dataset_path} "
    )
    subprocess.run(cmd.split(), check=True, cwd=tmp_path)


@pytest.mark.tensorflow
@pytest.mark.functional
def test_mnist_train_tf(tf_env, install_requirements, tmp_path):
    """
    Test MNIST tensorflow trainer by running it end-to-end.
    """
    install_requirements(TF_PATH, tf_env)
    conf_name = "pipeline"
    cmd = f"{tf_env}/bin/itwinai exec-pipeline --config-path {TF_PATH} \
        --config-name {conf_name} +pipe_key=pipeline"
    subprocess.run(cmd.split(), check=True, cwd=tmp_path)
