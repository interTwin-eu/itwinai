from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from itwinai.slurm.slurm_script_builder import (
    SlurmScriptBuilder,
    SlurmScriptConfiguration,
)


@pytest.fixture
def slurm_config_with_all_fields(tmp_path) -> SlurmScriptConfiguration:
    """Returns a SLURM script configuration with values for all field, including optional
    ones.
    """
    return SlurmScriptConfiguration(
        job_name="test_job",
        account="test_account",
        partition="test_partition",
        time="01:00:00",
        std_out=tmp_path / "my_test_job.out",
        err_out=tmp_path / "my_test_job.err",
        num_nodes=1,
        num_tasks_per_node=4,
        gpus_per_node=2,
        cpus_per_task=16,
        pre_exec_command="source .venv/bin/activate",
        memory="16G",
        exec_command="python main.py",
    )


def test_slurm_script_directives(
    tmp_path: Path, slurm_config_with_all_fields: SlurmScriptConfiguration
):
    """Checks that all directives exist in the saved script"""
    assert isinstance(slurm_config_with_all_fields.std_out, Path)
    assert isinstance(slurm_config_with_all_fields.err_out, Path)

    save_dir = tmp_path
    slurm_script_builder = SlurmScriptBuilder(
        slurm_script_configuration=slurm_config_with_all_fields,
        should_submit=False,
        should_save=True,
        save_dir=save_dir,
    )
    slurm_script_builder.process_script()

    assert save_dir.exists()
    save_path = save_dir / f"{slurm_script_builder.slurm_script_configuration.job_name}.slurm"

    with open(save_path) as f:
        file = f.read()

    directives = [
        f"#SBATCH --job-name={slurm_config_with_all_fields.job_name}",
        f"#SBATCH --account={slurm_config_with_all_fields.account}",
        f"#SBATCH --partition={slurm_config_with_all_fields.partition}",
        f"#SBATCH --time={slurm_config_with_all_fields.time}",
        f"#SBATCH --output={slurm_config_with_all_fields.std_out.as_posix()}",
        f"#SBATCH --error={slurm_config_with_all_fields.err_out.as_posix()}",
        f"#SBATCH --nodes={slurm_config_with_all_fields.num_nodes}",
        f"#SBATCH --ntasks-per-node={slurm_config_with_all_fields.num_tasks_per_node}",
        f"#SBATCH --cpus-per-task={slurm_config_with_all_fields.cpus_per_task}",
        f"#SBATCH --gpus-per-node={slurm_config_with_all_fields.gpus_per_node}",
        f"#SBATCH --gres=gpu:{slurm_config_with_all_fields.gpus_per_node}",
    ]
    for directive in directives:
        assert directive in file


@pytest.mark.parametrize(
    ("should_save_script", "should_submit_script"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_save_slurm_script(
    tmp_path: Path,
    slurm_config_with_all_fields: SlurmScriptConfiguration,
    should_save_script: bool,
    should_submit_script: bool,
):
    save_dir = tmp_path
    slurm_script_builder = SlurmScriptBuilder(
        slurm_script_configuration=slurm_config_with_all_fields,
        should_submit=should_submit_script,
        should_save=should_save_script,
        save_dir=save_dir,
    )
    with patch("subprocess.run"):
        slurm_script_builder.process_script()

    if should_save_script:
        assert save_dir.exists()


@pytest.mark.parametrize(
    ("should_save_script", "should_submit_script"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_submit_slurm_script(
    tmp_path: Path,
    slurm_config_with_all_fields: SlurmScriptConfiguration,
    should_save_script: bool,
    should_submit_script: bool,
):
    save_dir = tmp_path
    slurm_script_builder = SlurmScriptBuilder(
        slurm_script_configuration=slurm_config_with_all_fields,
        should_submit=should_submit_script,
        should_save=should_save_script,
        save_dir=save_dir,
    )
    with patch("subprocess.run") as mock_run:
        slurm_script_builder.process_script()
        if should_submit_script:
            mock_run.assert_called_once()
            called_args = mock_run.call_args[0][0]
            assert called_args[0] == "sbatch"
        else:
            mock_run.assert_not_called()


def test_save_slurm_script_twice(
    tmp_path: Path,
    slurm_config_with_all_fields: SlurmScriptConfiguration,
):
    save_dir = tmp_path
    slurm_script_builder = SlurmScriptBuilder(
        slurm_script_configuration=slurm_config_with_all_fields,
        should_submit=False,
        should_save=True,
        save_dir=save_dir,
    )
    slurm_script_builder.process_script()
    with pytest.raises(typer.Exit):
        slurm_script_builder.process_script()


def test_submit_slurm_script_twice(
    tmp_path: Path,
    slurm_config_with_all_fields: SlurmScriptConfiguration,
):
    save_dir = tmp_path
    slurm_script_builder = SlurmScriptBuilder(
        slurm_script_configuration=slurm_config_with_all_fields,
        should_submit=True,
        should_save=False,
        save_dir=save_dir,
    )
    with patch("subprocess.run"):
        slurm_script_builder.process_script()
        slurm_script_builder.process_script()
