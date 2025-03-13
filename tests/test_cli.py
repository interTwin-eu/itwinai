import inspect

from itwinai.cli import generate_slurm
from itwinai.slurm.utils import get_slurm_job_parser


def test_cli_slurm_function_signature():
    """Test that function signature in cli.py matches argparser"""
    args = inspect.getfullargspec(generate_slurm).args
    parser = get_slurm_job_parser()

    ignored_args = ["print_config", "help"]
    parser_args = {arg.dest for arg in parser._actions}
    parser_args -= set(ignored_args)

    missing_in_function = parser_args - set(args)
    missing_in_parser = set(args) - parser_args

    assert not missing_in_function and not missing_in_parser, (
        f"Arguments missing in function: {missing_in_function}, "
        f"Arguments missing in parser: {missing_in_parser}"
    )
