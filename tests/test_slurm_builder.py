import os
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest

from itwinai.slurm.slurm_script_builder import (
    SlurmScriptBuilder,
    SlurmScriptConfiguration,
)


@pytest.fixture
def cd_tmpdir(tmp_path) -> Generator[None, None, None]:
    old_cwd = Path.cwd()
    os.chdir(tmp_path)
    yield
    os.chdir(old_cwd)


@pytest.fixture
def slurm_config():
    return SlurmScriptConfiguration(
        job_name="test_job",
        account="test_account",
        partition="test_partition",
        time="01:00:00",
        std_out=Path("slurm_job_logs/test_job.out"),
        err_out=Path("slurm_job_logs/test_job.err"),
        num_nodes=2,
        num_tasks_per_node=4,
        gpus_per_node=2,
        cpus_per_task=16,
    )


@pytest.fixture
def slurm_builder(slurm_config: SlurmScriptConfiguration):
    return SlurmScriptBuilder(
        slurm_script_configuration=slurm_config,
        distributed_strategy="ddp",
        training_command="python train.py",
    )


@pytest.mark.parametrize(
    "save_script, submit_slurm_job",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_process_slurm_script(
    slurm_builder: SlurmScriptBuilder,
    save_script: bool,
    submit_slurm_job: bool,
    tmp_path: Path,
    cd_tmpdir: None,
):
    """Test that process_slurm_script behaves correctly for all cases of save_script
    and submit_slurm_job."""

    file_path = tmp_path / "slurm_script.sh"

    with patch("subprocess.run") as mock_run:
        slurm_builder.process_slurm_script(
            file_path=file_path,
            save_script=save_script,
            submit_slurm_job=submit_slurm_job,
        )

        # Checking that it creates the stdout and stderr directories

        assert slurm_builder.slurm_script_configuration.std_out is not None
        assert slurm_builder.slurm_script_configuration.err_out is not None
        if submit_slurm_job:
            assert slurm_builder.slurm_script_configuration.std_out.parent.exists()
            assert slurm_builder.slurm_script_configuration.err_out.parent.exists()

        # Check if sbatch was called
        if submit_slurm_job:
            mock_run.assert_called_once()
            called_args = mock_run.call_args[0][0]
            assert called_args[0] == "sbatch"
        else:
            mock_run.assert_not_called()

        # Check if the script file exists
        assert file_path.exists() == save_script


def test_process_slurm_script_twice(
    slurm_builder: SlurmScriptBuilder, tmp_path: Path, cd_tmpdir: None
):
    """Ensure that calling process_slurm_script twice with save_script=True fails,
    but calling it first with save_script=True and then with save_script=False works fine.
    """

    file_path = Path(tmp_path) / "slurm_script.sh"

    # First call with save_script=True should succeed
    slurm_builder.process_slurm_script(
        file_path=file_path, save_script=True, submit_slurm_job=False
    )
    assert file_path.exists()  # The script should be saved

    # Second call with save_script=False should not fail
    slurm_builder.process_slurm_script(
        file_path=file_path, save_script=False, submit_slurm_job=False
    )
    assert file_path.exists()  # The script should still be there

    # Second call with save_script=True should fail due to file already existing
    with pytest.raises(FileExistsError):
        slurm_builder.process_slurm_script(
            file_path=file_path, save_script=True, submit_slurm_job=False
        )
