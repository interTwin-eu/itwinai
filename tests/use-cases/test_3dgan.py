"""
Tests for CERN use case (3DGAN).
"""
import pytest
import subprocess
import os

CERN_PATH = "use-cases/3dgan"
CKPT_NAME = "3dgan-inference.pth"
DEFAULT_DATASET_PATH = 'exp_data'


@pytest.mark.skip("deprecated")
def test_structure_3dgan(check_folder_structure):
    """Test 3DGAN folder structure."""
    check_folder_structure(CERN_PATH)


@pytest.mark.functional
def test_3dgan_train(torch_env, tmp_test_dir, install_requirements):
    """
    Test 3DGAN torch lightning trainer by running it end-to-end.

    If CERN_DATASET env variable is defined, it is used to
    override the default dataset download location: useful
    when it contains a local copy of the dataset, preventing
    downloading it again.
    """
    install_requirements(CERN_PATH, torch_env)
    if os.environ.get('CERN_DATASET'):
        dataset_path = os.environ.get('CERN_DATASET')
    else:
        dataset_path = DEFAULT_DATASET_PATH
    conf = os.path.join(os.path.abspath(CERN_PATH), 'config.yaml')
    cmd = (f"{torch_env}/bin/itwinai exec-pipeline "
           f"--config {conf} --pipe-key training_pipeline "
           f'-o dataset_location={dataset_path} '
           '-o hw_accelerators=auto '
           '-o distributed_strategy=auto '
           '-o mlflow_tracking_uri=ml_logs/mlflow'
           )
    subprocess.run(cmd.split(), check=True, cwd=tmp_test_dir)


@pytest.mark.functional
def test_3dgan_inference(
    torch_env,
    tmp_test_dir,
    install_requirements,
    # fake_model_checkpoint
):
    """
    Test 3DGAN torch lightning trainer by running it end-to-end.

    If CERN_DATASET env variable is defined, it is used to
    override the default dataset download location: useful
    when it contains a local copy of the dataset, preventing
    downloading it again.
    """
    install_requirements(CERN_PATH, torch_env)

    # Create fake inference dataset and checkpoint
    exec = os.path.join(os.path.abspath(CERN_PATH),
                        'create_inference_sample.py')
    cmd = (f"{torch_env}/bin/python {exec} "
           f"--root {tmp_test_dir} "
           f"--ckpt-name {CKPT_NAME}")
    subprocess.run(cmd.split(), check=True, cwd=tmp_test_dir)

    # Test inference
    if os.environ.get('CERN_DATASET'):
        dataset_path = os.environ.get('CERN_DATASET')
    else:
        dataset_path = DEFAULT_DATASET_PATH
    conf = os.path.join(os.path.abspath(CERN_PATH), 'config.yaml')
    cmd = (
        f'{torch_env}/bin/itwinai exec-pipeline '
        f'--config {conf} --pipe-key inference_pipeline '
        f'-o dataset_location={dataset_path} '
        f'-o inference_model_uri={CKPT_NAME} '
        '-o hw_accelerators=auto '
        '-o distributed_strategy=auto '
        '-o logs_dir=ml_logs/mlflow_logs '
        '-o inference_results_location=3dgan-generated-data '
    )
    subprocess.run(cmd.split(), check=True, cwd=tmp_test_dir)
