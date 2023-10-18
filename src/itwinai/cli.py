"""
Command line interface for out Python application.
You can call commands from the command line.
Example

>>> $ itwinai --help

"""

# NOTE: import libs in the command"s function, not here.
# Otherwise this will slow the whole CLI.

import typer


app = typer.Typer()


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
