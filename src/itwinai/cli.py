"""
Command line interface for out Python application.
You can call commands from the command line.
Example

>>> $ itwinai --help

"""

# NOTE: import libs in the command"s function, not here.
# Otherwise this will slow the whole CLI.

from typing import Optional, List
from typing_extensions import Annotated
from pathlib import Path
import typer


app = typer.Typer()


@app.command()
def exec_pipeline(
    config: Annotated[Path, typer.Option(
        help="Path to the configuration file of the pipeline to execute."
    )],
    pipe_key: Annotated[str, typer.Option(
        help=("Key in the configuration file identifying "
              "the pipeline object to execute.")
    )] = "pipeline",
    overrides_list: Annotated[
        Optional[List[str]], typer.Option(
            "--override", "-o",
            help=(
                "Nested key to dynamically override elements in the "
                "configuration file with the "
                "corresponding new value, joined by '='. It is also possible "
                "to index elements in lists using their list index. "
                "Example: [...] "
                "-o pipeline.init_args.trainer.init_args.lr=0.001 "
                "-o pipeline.my_list.2.batch_size=64 "
            )
        )
    ] = None
):
    """Execute a pipeline from configuration file.
    Allows dynamic override of fields.
    """
    # Add working directory to python path so that the interpreter is able
    # to find the local python files imported from the pipeline file
    import os
    import sys

    sys.path.append(os.path.dirname(config))
    sys.path.append(os.getcwd())

    # Parse and execute pipeline
    from itwinai.parser import ConfigParser
    overrides = {
        k: v for k, v
        in map(lambda x: (x.split('=')[0], x.split('=')[1]), overrides_list)
    }
    parser = ConfigParser(config=config, override_keys=overrides)
    pipeline = parser.parse_pipeline(pipeline_nested_key=pipe_key)
    pipeline.execute()


@app.command()
def mlflow_ui(
    path: str = typer.Option("ml-logs/", help="Path to logs storage."),
):
    """
    Visualize Mlflow logs.
    """
    import subprocess

    subprocess.run(f"mlflow ui --backend-store-uri {path}".split())


@app.command()
def mlflow_server(
    path: str = typer.Option("ml-logs/", help="Path to logs storage."),
    port: int = typer.Option(
        5000, help="Port on which the server is listening."),
):
    """
    Spawn Mlflow server.
    """
    import subprocess

    subprocess.run(
        f"mlflow server --backend-store-uri {path} --port {port}".split())


@app.command()
def kill_mlflow_server(
    port: int = typer.Option(
        5000, help="Port on which the server is listening."),
):
    """
    Kill Mlflow server.
    """
    import subprocess

    subprocess.run(
        f"kill -9 $(lsof -t -i:{port})",
        shell=True,
        check=True,
        stderr=subprocess.DEVNULL
    )


if __name__ == "__main__":
    app()
