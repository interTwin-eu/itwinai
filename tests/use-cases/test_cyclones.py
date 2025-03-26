# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------
"""Tests for Cyclones use case.

Intended to be integration tests, to make sure that updates in the code base
do not break use cases' workflows.
"""

import os
import subprocess
from pathlib import Path

import pytest

CYCLONES_PATH = Path("use-cases", "cyclones")


@pytest.mark.skip("deprecated")
def test_structure_cyclones(check_folder_structure):
    """Test cyclones folder structure."""
    check_folder_structure(CYCLONES_PATH)


@pytest.mark.skip("deprecated")
@pytest.mark.functional
@pytest.mark.memory_heavy
def test_cyclones_train_tf(tf_env, install_requirements, tmp_path):
    """
    Test Cyclones tensorflow trainer by running it end-to-end.

    If CMCCC_DATASET env variable is defined, it is used to
    override the default dataset download location: useful
    when it contains a local copy of the dataset, preventing
    downloading it again.
    """
    # TODO: create a small sample dataset for tests only
    install_requirements(CYCLONES_PATH, tf_env)

    dataset_path = os.environ.get("CMCCC_DATASET", "./data/tmp_data")
    pipe = CYCLONES_PATH / "pipeline.yaml"
    train = CYCLONES_PATH / "train.py"

    cmd = (
        f"{tf_env}/bin/python {train.resolve()} -p {pipe.resolve()} --data_path {dataset_path}"
    )
    subprocess.run(cmd.split(), check=True, cwd=tmp_path)
