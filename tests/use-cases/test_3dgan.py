"""
Tests for CERN use case (3DGAN).
"""
import pytest
import subprocess
import os

CERN_PATH = "use-cases/3dgan"
CKPT_PATH = "3dgan-inference.pth"


@pytest.fixture(scope="module")
def fake_model_checkpoint() -> None:
    """
    Create a dummy model checkpoint for inference
    under ``CERN_PATH`` location.
    """
    import sys
    import torch
    curr_path = os.getcwd()
    os.chdir(CERN_PATH)
    sys.path.append(os.getcwd())

    from model import ThreeDGAN
    net = ThreeDGAN()
    torch.save(net, CKPT_PATH)

    sys.path.pop(sys.path.index(os.getcwd()))
    os.chdir(curr_path)


@pytest.mark.skip("deprecated")
def test_structure_3dgan(check_folder_structure):
    """Test 3DGAN folder structure."""
    check_folder_structure(CERN_PATH)


@pytest.mark.functional
def test_3dgan_train(torch_env, install_requirements):
    """
    Test 3DGAN torch lightning trainer by running it end-to-end.
    """
    install_requirements(CERN_PATH, torch_env)
    trainer_params = "pipeline.init_args.steps.training_step.init_args"
    cmd = (f"{torch_env}/bin/itwinai exec-pipeline "
           f"--config pipeline.yaml "
           f'-o {trainer_params}.config.trainer.accelerator=cpu '
           f'-o {trainer_params}.config.trainer.strategy=auto '
           )
    subprocess.run(cmd.split(), check=True, cwd=CERN_PATH)


@pytest.mark.functional
def test_3dgan_inference(
    torch_env,
    install_requirements,
    fake_model_checkpoint
):
    """
    Test 3DGAN torch lightning trainer by running it end-to-end.
    """
    install_requirements(CERN_PATH, torch_env)
    # cmd = (f"micromamba run -p {torch_env} python "
    #        f"{CERN_PATH}/train.py -p {CERN_PATH}/pipeline.yaml")
    # cmd = (f"micromamba run -p {torch_env} itwinai exec-pipeline "
    #        f"--config {CERN_PATH}/inference-pipeline.yaml")

    getter_params = "pipeline.init_args.steps.dataloading_step.init_args"
    trainer_params = "pipeline.init_args.steps.inference_step.init_args"
    logger_params = trainer_params + ".config.trainer.logger.init_args"
    data_params = trainer_params + ".config.data.init_args"
    saver_params = "pipeline.init_args.steps.saver_step.init_args"
    cmd = (
        f'{torch_env}/bin/itwinai exec-pipeline '
        '--config inference-pipeline.yaml '
        f'-o {getter_params}.data_path=exp_data '
        f'-o {trainer_params}.model.init_args.model_uri={CKPT_PATH} '
        f'-o {trainer_params}.config.trainer.accelerator=cpu '
        f'-o {trainer_params}.config.trainer.strategy=auto '
        f'-o {logger_params}.save_dir=ml_logs/mlflow_logs '
        f'-o {data_params}.datapath=exp_data/*/*.h5 '
        f'-o {saver_params}.save_dir=3dgan-generated-data '
    )
    subprocess.run(cmd.split(), check=True, cwd=CERN_PATH)
