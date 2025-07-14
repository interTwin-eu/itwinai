# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Alex Krochak
#
# Credit:
# - Alex Krochak <o.krochak@fz-juelich.de> - FZJ
# --------------------------------------------------------------------------------------

"""Tests for radio-astronomy use case.

Intended to be integration tests, to make sure that updates in the code base
do not break use cases' workflows.

This is meant to be run from the main itwinai directory, not the use-case folder !!!
"pytest use-cases/radio-astronomy/tests/test_radio-astronomy.py"

NOTE FOR DEVELOPERS: if you are editing this file, make sure that entries in 
use-cases/radio-astronomy/.config-test.yaml are updated accordingly !!! 
"""

import os
import subprocess
from pathlib import Path
import shutil

import pytest

USECASE_FOLDER = Path("use-cases", "radio-astronomy").resolve()

@pytest.fixture
def torch_env() -> str:
    """Returns absolute path to torch virtual environment."""
    env_path = Path(os.environ.get("TORCH_ENV", "./.venv"))
    return str(env_path.resolve())

@pytest.fixture
def syndata(tmp_path, torch_env,install_requirements):
    # This fixture implicitly tests the synthetic data generation pipeline
    install_requirements(USECASE_FOLDER, torch_env)

    cmd_data = (
        f"{torch_env}/bin/itwinai exec-pipeline --config-name .config-test "
        f"+pipe_key=syndata_pipeline ++syndata_test_dir={tmp_path}/ "
    )
    if len(os.listdir(tmp_path)) == 0:  # only run if directory is empty
        # Copy the necessary files to the temporary directory for testing
        shutil.copy(USECASE_FOLDER / ".config-test.yaml", tmp_path)
        shutil.copy(USECASE_FOLDER / "data.py", tmp_path)
        shutil.copy(USECASE_FOLDER / "trainer.py", tmp_path)

        print(f"Running synthetic data generation command: {cmd_data}")
        
        subprocess.run(cmd_data.split(), check=True, cwd=tmp_path)

    return tmp_path

@pytest.fixture
def generate_unet(torch_env, syndata):
    """Generate the U-Net model for the Filter-CNN test. """
    cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline --config-name .config-test "
        f"+pipe_key=unet_pipeline ++image_directory={syndata}/ ++mask_directory={syndata}/ "
    )

    subprocess.run(cmd.split(), check=True, cwd=syndata)

@pytest.mark.skip(reason="not enough resources in the CI container to run this test")
def test_radio_astronomy_unet(torch_env, syndata, install_requirements):
    """Test U-Net Pulsar-DDT trainer by running it end-to-end
    via the config-test.yaml configuration file."""

    install_requirements(USECASE_FOLDER, torch_env)

    cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline --config-name .config-test "
        f"+pipe_key=unet_pipeline ++image_directory={syndata}/ ++mask_directory={syndata}/ "
    )

    subprocess.run(cmd.split(), check=True, cwd=syndata)

# @pytest.mark.functional
@pytest.mark.skip(reason="not enough resources in the CI container to run this test")
def test_radio_astronomy_filtercnn(torch_env, syndata, generate_unet, install_requirements):
    """Test Filter-CNN Pulsar-DDT trainer by running it end-to-end
    via the config-test.yaml configuration file. Requires the U-Net model to be present."""

    install_requirements(USECASE_FOLDER, torch_env)

    cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline --config-name .config-test "
        f"+pipe_key=fcnn_pipeline ++image_directory={syndata}/ ++mask_directory={syndata}/ "
    )

    subprocess.run(cmd.split(), check=True, cwd=syndata)

@pytest.mark.skip(reason="not enough resources in the CI container to run this test")
def test_radio_astronomy_cnn1d(torch_env, syndata, install_requirements):
    """Test CNN-1D Pulsar-DDT trainer by running it end-to-end
    via the config-test.yaml configuration file."""

    install_requirements(USECASE_FOLDER, torch_env)

    cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline --config-name .config-test "
        f"+pipe_key=cnn1d_pipeline ++image_directory={syndata}/ ++mask_directory={syndata}/ "
    )

    subprocess.run(cmd.split(), check=True, cwd=syndata)

@pytest.mark.skip(reason="Requires all moodels to be saved to disk.")
def test_radio_astronomy_evaluate(torch_env):
    """Test the evaluate pipeline by running it end-to-end
    via the config-test.yaml configuration file."""

    cmd = (
        f"{torch_env}/bin/itwinai exec-pipeline "
        f"--config-name .config-test "
        f"+pipe_key=evaluate_pipeline "
    )

    ## Run the pipeline and check file generation in the use-case folder
    subprocess.run(cmd.split(), check=True, cwd=USECASE_FOLDER)
    ## Clean up the use-case folder
    subprocess.run("./.pytest-clean", check=True, cwd=USECASE_FOLDER)
