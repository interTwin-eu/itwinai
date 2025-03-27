# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
# Created by: Oleksandr Krochak
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

USECASE_FOLDER = Path("use-cases", "radio-astronomy").resolve()
# USECASE_FOLDER = Path().resolve()

DEFAULT_MNIST_DATASET = ".tmp"



@pytest.fixture
def torch_env() -> str:
    """If torch_env env variable is defined, it overrides the default
    torch virtual environment name. Otherwise, fall back
    to './.venv-pytorch'.

    Returns absolute path to torch virtual environment.
    """
    return "/p/project1/intertwin/krochak1/itwinai/.venv"

@pytest.mark.functional
def test_radio_astronomy_train(torch_env):
    """
    Test MNIST torch native trainer by running it end-to-end.

    If MNIST_DATASET env variable is defined, it is used to
    override the default dataset download location: useful
    when it contains a local copy of the dataset, preventing
    downloading it again.
    """
    # install_requirements(TORCH_PATH, torch_env)

    # dataset_path = os.environ.get("MNIST_DATASET", DEFAULT_MNIST_DATASET)

    cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline "
        f"--config-path {USECASE_FOLDER} "
        f"--config-name config-test "
        f"+pipe_key=unet_pipeline "
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(cmd.split(), check=True)


# @pytest.mark.functional
# def test_mnist_inference_torch(torch_env, install_requirements):
#     """
#     Test MNIST torch native inference by running it end-to-end.
#     """
#     install_requirements(TORCH_PATH, torch_env)

#     exec = TORCH_PATH / "create_inference_sample.py"

#     run_inference_cmd = f"{torch_env}/bin/itwinai exec-pipeline --config-path {TORCH_PATH} \
#         +pipe_key=inference_pipeline"
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Create fake inference dataset and checkpoint
#         generate_model_cmd = f"{torch_env}/bin/python {exec} --root {temp_dir}"
#         subprocess.run(generate_model_cmd.split(), check=True, cwd=temp_dir)

#         # Running inference
#         subprocess.run(run_inference_cmd.split(), check=True, cwd=temp_dir)


# @pytest.mark.functional
# def test_mnist_train_torch_lightning(torch_env, install_requirements):
#     """
#     Test MNIST torch lightning trainer by running it end-to-end.

#     If MNIST_DATASET env variable is defined, it is used to
#     override the default dataset download location: useful
#     when it contains a local copy of the dataset, preventing
#     downloading it again.
#     """
#     install_requirements(LIGHTNING_PATH, torch_env)

#     dataset_path = os.environ.get("MNIST_DATASET", DEFAULT_MNIST_DATASET)
#     cmd = (
#         f"{torch_env}/bin/itwinai exec-pipeline "
#         f"--config-path {LIGHTNING_PATH} "
#         f"dataset_root={dataset_path} "
#     )
#     with tempfile.TemporaryDirectory() as temp_dir:
#         subprocess.run(cmd.split(), check=True, cwd=temp_dir)


# @pytest.mark.tensorflow
# @pytest.mark.functional
# def test_mnist_train_tf(tf_env, install_requirements):
#     """
#     Test MNIST tensorflow trainer by running it end-to-end.
#     """
#     install_requirements(TF_PATH, tf_env)
#     conf_name = "pipeline"
#     cmd = f"{tf_env}/bin/itwinai exec-pipeline --config-path {TF_PATH} \
#         --config-name {conf_name} +pipe_key=pipeline"
#     with tempfile.TemporaryDirectory() as temp_dir:
#         subprocess.run(cmd.split(), check=True, cwd=temp_dir)
