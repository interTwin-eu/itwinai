"""
Tests for CERN use case (3DGAN).
"""
import pytest
import subprocess
from itwinai.utils import dynamically_import_class

CERN_PATH = "use-cases/3dgan"
CKPT_PATH = "3dgan-inference.pth"


@pytest.fixture(scope="module")
def fake_model_checkpoint() -> None:
    """
    Create a dummy model checkpoint for inference.
    """
    import sys
    import torch
    sys.path.append(CERN_PATH)
    from model import ThreeDGAN
    ThreeDGAN = dynamically_import_class('model.ThreeDGAN')
    net = ThreeDGAN()
    torch.save(net, CKPT_PATH)


def test_structure_3dgan(check_folder_structure):
    """Test 3DGAN folder structure."""
    check_folder_structure(CERN_PATH)


@pytest.mark.functional
def test_3dgan_train(install_requirements):
    """
    Test 3DGAN torch lightning trainer by running it end-to-end.
    """
    install_requirements(CERN_PATH, pytest.TORCH_PREFIX)
    # cmd = (f"micromamba run -p {pytest.TORCH_PREFIX} python "
    #        f"{CERN_PATH}/train.py -p {CERN_PATH}/pipeline.yaml")
    cmd = (f"micromamba run -p {pytest.TORCH_PREFIX} itwinai exec-pipeline "
           f"--config {CERN_PATH}/pipeline.yaml")
    subprocess.run(cmd.split(), check=True)


@pytest.mark.functional
def test_3dgan_inference(install_requirements, fake_model_checkpoint):
    """
    Test 3DGAN torch lightning trainer by running it end-to-end.
    """
    install_requirements(CERN_PATH, pytest.TORCH_PREFIX)
    # cmd = (f"micromamba run -p {pytest.TORCH_PREFIX} python "
    #        f"{CERN_PATH}/train.py -p {CERN_PATH}/pipeline.yaml")
    # cmd = (f"micromamba run -p {pytest.TORCH_PREFIX} itwinai exec-pipeline "
    #        f"--config {CERN_PATH}/inference-pipeline.yaml")

    getter_params = "pipeline.init_args.steps.0.init_args"
    trainer_params = "pipeline.init_args.steps.1.init_args"
    logger_params = trainer_params + ".config.trainer.logger.init_args"
    data_params = trainer_params + ".config.data.init_args"
    saver_params = "pipeline.init_args.steps.2.init_args"
    cmd = (
        'itwinai exec-pipeline '
        '--config use-cases/3dgan/inference-pipeline.yaml '
        f'-o {getter_params}.data_path=exp_data '
        f'-o {trainer_params}.model.init_args.model_uri="{CKPT_PATH}" '
        f'-o {logger_params}.save_dir=ml_logs/mlflow_logs '
        f'-o {data_params}.datapath="exp_data/*/*.h5" '
        f'-o {saver_params}.save_dir=3dgan-generated-data '
    )
    subprocess.run(cmd.split(), check=True)
