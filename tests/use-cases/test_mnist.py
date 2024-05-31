"""
Tests for MNIST use case.

Intended to be integration tests, to make sure that updates in the code base
do not break use cases' workflows.
"""

import pytest
import subprocess
import os
# from itwinai.cli import exec_pipeline

TORCH_PATH = "use-cases/mnist/torch"
LIGHTNING_PATH = "use-cases/mnist/torch-lightning"
TF_PATH = "use-cases/mnist/tensorflow"
DEFAULT_MNIST_DATASET = '.tmp'


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
def test_mnist_train_torch(torch_env, tmp_test_dir, install_requirements):
    """
    Test MNIST torch native trainer by running it end-to-end.

    If MNIST_DATASET env variable is defined, it is used to
    override the default dataset download location: useful
    when it contains a local copy of the dataset, preventing
    downloading it again.
    """
    install_requirements(TORCH_PATH, torch_env)

    if os.environ.get('MNIST_DATASET'):
        dataset_path = os.environ.get('MNIST_DATASET')
    else:
        dataset_path = DEFAULT_MNIST_DATASET
    conf = os.path.join(os.path.abspath(TORCH_PATH), 'config.yaml')
    cmd = (f"{torch_env}/bin/itwinai exec-pipeline "
           f"--config {conf} --pipe-key training_pipeline "
           f"-o dataset_root={dataset_path}")
    subprocess.run(cmd.split(), check=True, cwd=tmp_test_dir)


@pytest.mark.functional
def test_mnist_inference_torch(torch_env, tmp_test_dir, install_requirements):
    """
    Test MNIST torch native inference by running it end-to-end.
    """
    install_requirements(TORCH_PATH, torch_env)

    # Create fake inference dataset and checkpoint
    exec = os.path.join(os.path.abspath(TORCH_PATH),
                        'create_inference_sample.py')
    cmd = (f"{torch_env}/bin/python {exec} "
           f"--root {tmp_test_dir}")
    subprocess.run(cmd.split(), check=True, cwd=tmp_test_dir)

    # Test inference
    conf = os.path.join(os.path.abspath(TORCH_PATH), 'config.yaml')
    cmd = (f"{torch_env}/bin/itwinai exec-pipeline "
           f"--config {conf} --pipe-key inference_pipeline")
    subprocess.run(cmd.split(), check=True, cwd=tmp_test_dir)


@pytest.mark.functional
def test_mnist_train_torch_lightning(
    torch_env,
    tmp_test_dir,
    install_requirements
):
    """
    Test MNIST torch lightning trainer by running it end-to-end.

    If MNIST_DATASET env variable is defined, it is used to
    override the default dataset download location: useful
    when it contains a local copy of the dataset, preventing
    downloading it again.
    """
    install_requirements(LIGHTNING_PATH, torch_env)

    if os.environ.get('MNIST_DATASET'):
        dataset_path = os.environ.get('MNIST_DATASET')
    else:
        dataset_path = DEFAULT_MNIST_DATASET
    conf = os.path.join(os.path.abspath(LIGHTNING_PATH), 'config.yaml')
    cmd = (f"{torch_env}/bin/itwinai exec-pipeline "
           f"--config {conf} --pipe-key training_pipeline "
           f"-o dataset_root={dataset_path}")
    subprocess.run(cmd.split(), check=True, cwd=tmp_test_dir)


@pytest.mark.functional
def test_mnist_train_tf(tf_env, tmp_test_dir, install_requirements):
    """
    Test MNIST tensorflow trainer by running it end-to-end.
    """
    install_requirements(TF_PATH, tf_env)
    conf = os.path.join(os.path.abspath(TF_PATH), 'pipeline.yaml')
    cmd = (f"{tf_env}/bin/itwinai exec-pipeline "
           f"--config {conf} --pipe-key pipeline")
    subprocess.run(cmd.split(), check=True, cwd=tmp_test_dir)
