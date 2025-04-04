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

"""Tests for CERN use case (3DGAN)."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

CERN_PATH = Path("use-cases", "3dgan").resolve()
CKPT_NAME = "3dgan-inference.pth"
DEFAULT_DATASET_PATH = "exp_data"


@pytest.mark.skip("deprecated")
def test_structure_3dgan(check_folder_structure):
    """Test 3DGAN folder structure."""
    check_folder_structure(CERN_PATH)


@pytest.mark.functional
def test_3dgan_train(torch_env, install_requirements, tmp_path):
    """Test 3DGAN torch lightning trainer by running it end-to-end.

    If CERN_DATASET env variable is defined, it is used to
    override the default dataset download location: useful
    when it contains a local copy of the dataset, preventing
    downloading it again.
    """
    install_requirements(CERN_PATH, torch_env)
    dataset_path = os.environ.get("CERN_DATASET", DEFAULT_DATASET_PATH)
    cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline "
        f"--config-path {CERN_PATH} "
        f"dataset_location={dataset_path} "
        "hw_accelerators=auto "
        "distributed_strategy=auto "
        "mlflow_tracking_uri=ml_logs/mlflow"
    )

    subprocess.run(cmd.split(), check=True, cwd=tmp_path)


@pytest.mark.functional
def test_3dgan_inference(
    torch_env,
    install_requirements,
    tmp_path,
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

    exec = CERN_PATH / "create_inference_sample.py"

    run_inference_cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline "
        f"--config-path {CERN_PATH} +pipe_key=inference_pipeline "
        f"dataset_location={dataset_path} "
        f"inference_model_uri={CKPT_NAME} "
        "hw_accelerators=auto "
        "distributed_strategy=auto "
        "logs_dir=ml_logs/mlflow_logs "
        "inference_results_location=3dgan-generated-data "
    )

    # Create fake inference dataset and checkpoint
    generate_model_cmd = (
        f"{torch_env}/bin/python {exec} "
        f"--root {str(Path(tmp_path).resolve())} "
        f"--ckpt-name {CKPT_NAME}"
    )
    subprocess.run(generate_model_cmd.split(), check=True, cwd=tmp_path)

    # Running the inference
    subprocess.run(run_inference_cmd.split(), check=True, cwd=tmp_path)
