"""
Tests for CERN use case (3DGAN).
"""
import pytest
import subprocess
import os

CERN_PATH = "use-cases/3dgan"
CKPT_NAME = "3dgan-inference.pth"


@pytest.mark.skip("deprecated")
def test_structure_3dgan(check_folder_structure):
    """Test 3DGAN folder structure."""
    check_folder_structure(CERN_PATH)


@pytest.mark.functional
def test_3dgan_train(torch_env, tmp_test_dir, install_requirements):
    """
    Test 3DGAN torch lightning trainer by running it end-to-end.
    """
    install_requirements(CERN_PATH, torch_env)
    conf = os.path.join(os.path.abspath(CERN_PATH), 'pipeline.yaml')
    trainer_params = "pipeline.init_args.steps.training_step.init_args"
    cmd = (f"{torch_env}/bin/itwinai exec-pipeline "
           f"--config {conf} "
           f'-o {trainer_params}.config.trainer.accelerator=cpu '
           f'-o {trainer_params}.config.trainer.strategy=auto '
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
    conf = os.path.join(os.path.abspath(CERN_PATH), 'inference-pipeline.yaml')
    getter_params = "pipeline.init_args.steps.dataloading_step.init_args"
    trainer_params = "pipeline.init_args.steps.inference_step.init_args"
    logger_params = trainer_params + ".config.trainer.logger.init_args"
    data_params = trainer_params + ".config.data.init_args"
    saver_params = "pipeline.init_args.steps.saver_step.init_args"
    cmd = (
        f'{torch_env}/bin/itwinai exec-pipeline '
        f'--config {conf} '
        f'-o {getter_params}.data_path=exp_data '
        f'-o {trainer_params}.model.init_args.model_uri={CKPT_NAME} '
        f'-o {trainer_params}.config.trainer.accelerator=auto '
        f'-o {trainer_params}.config.trainer.strategy=auto '
        f'-o {logger_params}.save_dir=ml_logs/mlflow_logs '
        f'-o {data_params}.datapath=exp_data/*/*.h5 '
        f'-o {saver_params}.save_dir=3dgan-generated-data '
    )
    subprocess.run(cmd.split(), check=True, cwd=CERN_PATH)
