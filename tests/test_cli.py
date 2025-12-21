import inspect

from itwinai.cli import generate_slurm


def test_cli_slurm_function_signature():
    """generate_slurm exposes config plus submit/save overrides."""
    args = inspect.getfullargspec(generate_slurm).args
    assert args == ["config", "submit_job", "save_script"]
