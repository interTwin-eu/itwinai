"""
Tests for MNIST use case.

Intended to be integration tests, to make sure that updates in the code base
do not break use cases' workflows.
"""

import pytest
import os
import sys
import subprocess
# from itwinai.cli import exec_pipeline

TORCH_PATH = "use-cases/mnist/torch"
LIGHTNING_PATH = "use-cases/mnist/torch-lightning"
TF_PATH = "use-cases/mnist/tensorflow"


def mnist_torch_inference_files(
    root: str = '.',
    samples_path: str = 'mnist-sample-data/',
    model_name: str = 'mnist-pre-trained.pth'
):
    """Create sample dataset and fake model to test mnist
    inference workflow. Assumes to be run from
    the use case folder.

    Args:
        root (str, optional): where to create the files.
        Defaults to '.'.
    """
    sys.path = [os.getcwd()] + sys.path

    from dataloader import InferenceMNIST
    sample = os.path.join(root, samples_path)
    InferenceMNIST.generate_jpg_sample(sample, 10)

    import torch
    raise ValueError(sys.path)
    from model import Net
    dummy_nn = Net()
    mdl_ckpt = os.path.join(root, model_name)
    torch.save(dummy_nn, mdl_ckpt)

    sys.path = sys.path[1:]


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
def test_mnist_train_torch(torch_env, install_requirements):
    """
    Test MNIST torch native trainer by running it end-to-end.

    To set the torch env path set the ``TORCH_ENV`` env variable:

    >>> export TORCH_ENV="my_env"
    """
    install_requirements(TORCH_PATH, torch_env)
    cmd = (f"{torch_env}/bin/itwinai exec-pipeline "
           f"--config config.yaml --pipe-key training_pipeline")
    subprocess.run(cmd.split(), check=True, cwd=TORCH_PATH)


@pytest.mark.functional
def test_mnist_inference_torch(torch_env, install_requirements):
    """
    Test MNIST torch native inference by running it end-to-end.

    To set the torch env path set the ``TORCH_ENV`` env variable:

    >>> export TORCH_ENV="my_env"
    """
    install_requirements(TORCH_PATH, torch_env)

    samples_path: str = 'mnist-sample-data/'
    model_name: str = 'mnist-pre-trained.pth'
    root_path = os.getcwd()
    os.chdir(TORCH_PATH)
    # sys.path.append(os.path.join(os.getcwd(), TORCH_PATH))
    # sys.path.append(os.getcwd())
    try:
        mnist_torch_inference_files(
            samples_path=samples_path,
            model_name=model_name
        )
        # exec_pipeline(
        #     config='config.yaml',
        #     pipe_key='inference_pipeline',
        #     overrides_list=[
        #         f"predictions_dir={samples_path}",
        #         f"inference_model_mlflow_uri={model_name}"
        #     ]
        # )
    except Exception as e:
        raise e
    finally:
        os.chdir(root_path)
        # sys.path.pop(sys.path.index(os.getcwd()))
    cmd = (f"{torch_env}/bin/itwinai exec-pipeline "
           f"--config config.yaml --pipe-key inference_pipeline")
    subprocess.run(cmd.split(), check=True, cwd=TORCH_PATH)


@pytest.mark.functional
def test_mnist_train_torch_lightning(torch_env, install_requirements):
    """
    Test MNIST torch lightning trainer by running it end-to-end.

    To set the torch env path set the ``TORCH_ENV`` env variable:

    >>> export TORCH_ENV="my_env"
    """
    install_requirements(TORCH_PATH, torch_env)
    cmd = (f"{torch_env}/bin/itwinai exec-pipeline "
           f"--config config.yaml --pipe-key training_pipeline")
    subprocess.run(cmd.split(), check=True, cwd=LIGHTNING_PATH)


@pytest.mark.functional
def test_mnist_train_tf(tf_env, install_requirements):
    """
    Test MNIST tensorflow trainer by running it end-to-end.
    """
    install_requirements(TF_PATH, tf_env)
    cmd = (f"{tf_env}/bin/itwinai exec-pipeline "
           f"--config pipeline.yaml --pipe-key pipeline")
    subprocess.run(cmd.split(), check=True, cwd=TF_PATH)
