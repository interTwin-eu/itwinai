"""
Tests for MNIST use case.

Intended to be integration tests, to make sure that updates in the code base
do not break use cases' workflows.
"""

import pytest
import subprocess

CYCLONES_PATH = "use-cases/cyclones"


@pytest.mark.skip("deprecated")
def test_structure_cyclones(check_folder_structure):
    """Test cyclones folder structure."""
    check_folder_structure(CYCLONES_PATH)


@pytest.mark.functional
@pytest.mark.memory_heavy
def test_cyclones_train_tf(tf_env, install_requirements):
    """
    Test Cyclones tensorflow trainer by running it end-to-end.
    """
    # TODO: create a small sample dataset for tests only
    install_requirements(CYCLONES_PATH, tf_env)
    cmd = (f"{tf_env}/bin/python train.py "
           f"-p pipeline.yaml")
    subprocess.run(cmd.split(), check=True, cwd=CYCLONES_PATH)
