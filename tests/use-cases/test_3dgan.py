# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Tests for CERN use case (3DGAN)."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

CERN_PATH = Path("use-cases", "3dgan")
CKPT_NAME = "3dgan-inference.pth"
DEFAULT_DATASET_PATH = "exp_data"


@pytest.mark.skip("deprecated")
def test_structure_3dgan(check_folder_structure):
    """Test 3DGAN folder structure."""
    check_folder_structure(CERN_PATH)


@pytest.mark.functional
def test_3dgan_train(torch_env, install_requirements):
    """Test 3DGAN torch lightning trainer by running it end-to-end.

    If CERN_DATASET env variable is defined, it is used to
    override the default dataset download location: useful
    when it contains a local copy of the dataset, preventing
    downloading it again.
    """
    install_requirements(CERN_PATH, torch_env)
    dataset_path = os.environ.get("CERN_DATASET", DEFAULT_DATASET_PATH)
    conf = (CERN_PATH / "config.yaml").resolve()
    cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline "
        f"--config {conf} --pipe-key training_pipeline "
        f"-o dataset_location={dataset_path} "
        "-o hw_accelerators=auto "
        "-o distributed_strategy=auto "
        "-o mlflow_tracking_uri=ml_logs/mlflow"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(cmd.split(), check=True, cwd=temp_dir)


@pytest.mark.functional
def test_3dgan_inference(
    torch_env,
    install_requirements,
    # fake_model_checkpoint
):
    """Test 3DGAN torch lightning trainer by running it end-to-end.

    If CERN_DATASET env variable is defined, it is used to
    override the default dataset download location: useful
    when it contains a local copy of the dataset, preventing
    downloading it again.
    """
    install_requirements(CERN_PATH, torch_env)

    # Test inference
    dataset_path = os.environ.get("CERN_DATASET", DEFAULT_DATASET_PATH)

    conf = (CERN_PATH / "config.yaml").resolve()
    exec = (CERN_PATH / "create_inference_sample.py").resolve()

    run_inference_cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline "
        f"--config {conf} --pipe-key inference_pipeline "
        f"-o dataset_location={dataset_path} "
        f"-o inference_model_uri={CKPT_NAME} "
        "-o hw_accelerators=auto "
        "-o distributed_strategy=auto "
        "-o logs_dir=ml_logs/mlflow_logs "
        "-o inference_results_location=3dgan-generated-data "
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create fake inference dataset and checkpoint
        generate_model_cmd = (
            f"{torch_env}/bin/python {exec} "
            f"--root {str(Path(temp_dir).resolve())} "
            f"--ckpt-name {CKPT_NAME}"
        )
        subprocess.run(generate_model_cmd.split(), check=True, cwd=temp_dir)

        # Running the inference
        subprocess.run(run_inference_cmd.split(), check=True, cwd=temp_dir)
