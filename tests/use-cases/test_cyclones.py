"""
Tests for MNIST use case.

Intended to be integration tests, to make sure that updates in the code base
do not break use cases' workflows.
"""

import pytest
import subprocess
import os

CYCLONES_PATH = "use-cases/cyclones"


@pytest.mark.skip("deprecated")
def test_structure_cyclones(check_folder_structure):
    """Test cyclones folder structure."""
    check_folder_structure(CYCLONES_PATH)


@pytest.mark.functional
@pytest.mark.memory_heavy
def test_cyclones_train_tf(tf_env, tmp_test_dir, install_requirements):
    """
    Test Cyclones tensorflow trainer by running it end-to-end.

    If CMCCC_DATASET env variable is defined, it is used to
    override the default dataset download location: useful
    when it contains a local copy of the dataset, preventing
    downloading it again.
    """
    # TODO: create a small sample dataset for tests only
    install_requirements(CYCLONES_PATH, tf_env)
    if os.environ.get('CMCCC_DATASET'):
        dataset_path = os.environ.get('CMCCC_DATASET')
    else:
        dataset_path = './data/tmp_data'
    pipe = os.path.join(os.path.abspath(CYCLONES_PATH), 'pipeline.yaml')
    train = os.path.join(os.path.abspath(CYCLONES_PATH), 'train.py')
    cmd = (f"{tf_env}/bin/python {train} "
           f"-p {pipe} --data_path {dataset_path}")
    subprocess.run(cmd.split(), check=True, cwd=tmp_test_dir)
