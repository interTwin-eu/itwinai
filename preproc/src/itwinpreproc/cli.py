"""
Command line interface for out Python application.
You can call commands from the command line.

Example:
>>> $ itwinpreproc --help
>>> $ itwinpreproc setup-env --help

Example for env setup:
>>> $ itwinpreproc setup-env --file requirements.txt

"""
from typing import Optional
import sys
import typer
from itwinpreproc.env import install_deps_list, install_deps_file

app = typer.Typer()


@app.command()
def setup_env(
    deps: Optional[str] = typer.Option(
        None,
        help=("Comma separated requirements, each in pip format "
              "(e.g., 'SomePackage==1.0.4,SomeOtherPackage>=2.3.0'). "
              "If given, other arguments are ignored.")
    ),
    file: Optional[str] = typer.Option(
        None,
        help=("Path to requirements text file, in pip format "
              "(e.g., 'SomePackage==1.0.4)")
    )
):
    """Install custom dependencies in this Python env using pip.
    """
    # TODO: improve using conda/mamba, for better compatibility
    # with base env. Or give both options...
    if deps is not None:
        install_deps_list(deps.split(','))
    elif file is not None:
        install_deps_file(file)
    else:
        print("No dependencies given!", file=sys.stderr)
        raise ValueError

if __name__ == "__main__":
    app()
