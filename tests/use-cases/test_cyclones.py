"""
Tests for MNIST use case.

Intended to be integration tests, to make sure that updates in the code base
do not break use cases' workflows.
"""

import pytest
import subprocess

CYCLONES_PATH = "use-cases/cyclones"


def test_structure_cyclones(check_folder_structure):
    """Test cyclones folder structure."""
    check_folder_structure(CYCLONES_PATH)


@pytest.mark.functional
@pytest.mark.memory_heavy
def test_cyclones_train_tf(install_requirements):
    """
    Test Cyclones tensorflow trainer by running it end-to-end.
    """
    install_requirements(CYCLONES_PATH, pytest.TF_PREFIX)
    cmd = (f"micromamba run -p {pytest.TF_PREFIX} python "
           f"{CYCLONES_PATH}/train.py -p {CYCLONES_PATH}/pipeline.yaml")
    subprocess.run(cmd.split(), check=True)
