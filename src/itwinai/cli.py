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
def scalability_report(
    pattern: Annotated[str, typer.Option(
        help="Python pattern matching names of CSVs in sub-folders."
    )],
    plot_title: Annotated[Optional[str], typer.Option(
        help=("Plot name.")
    )] = None,
    skip_id: Annotated[Optional[int], typer.Option(
        help=("Skip epoch ID.")
    )] = None,
    archive: Annotated[Optional[str], typer.Option(
        help=("Archive path where to backup the data, WITHOUT EXTENSION.")
    )] = None,
):
    """
    Generate scalability report merging all CSVs containing epoch time
    records in sub-folders.

    Example:
    >>> itwinai scalability-report --pattern="^epoch.+\.csv$" --skip-id 0 \
    >>>     --plot-title "Some title" --archive folder/archive_name
    """
    # TODO: add max depth and path different from CWD
    import os
    import re
    import shutil
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    regex = re.compile(r'{}'.format(pattern))
    combined_df = pd.DataFrame()
    csv_files = []
    for root, _, files in os.walk(os.getcwd()):
        for file in files:
            if regex.match(file):
                fpath = os.path.join(root, file)
                csv_files.append(fpath)
                df = pd.read_csv(fpath)
                if skip_id is not None:
                    df = df.drop(df[df.epoch_id == skip_id].index)
                combined_df = pd.concat([combined_df, df])
    print("Merged CSV:")
    print(combined_df)

    avg_times = (
        combined_df
        .drop(columns='epoch_id')
        .groupby(['name', 'nodes'])
        .mean()
        .reset_index()
    )
    print("\nAvg over name and nodes:")
    print(avg_times.rename(columns=dict(time='avg(time)')))

    # fig, (sp_up_ax, eff_ax) = plt.subplots(1, 2, figsize=(12, 4))
    fig, sp_up_ax = plt.subplots(1, 1, figsize=(6, 4))
    if plot_title is not None:
        fig.suptitle(plot_title)

    for name in set(avg_times.name.values):
        df = avg_times[avg_times.name == name].drop(columns='name')

        # Debug
        # compute_time = [3791., 1884., 1011., 598.]
        # nodes = [1, 2, 4, 8]
        # d = {'nodes': nodes, 'time': compute_time}
        # df = pd.DataFrame(data=d)

        df["NGPUs"] = df["nodes"]*4
        # speedup
        df["Speedup - ideal"] = df["nodes"].astype(float)
        df["Speedup"] = df["time"].iloc[0] / df["time"]
        df["Nworkers"] = 1

        # efficiency
        df["Threadscaled Sim. Time / s"] = df["time"] * \
            df["nodes"] * df["Nworkers"]
        df["Efficiency"] = df["Threadscaled Sim. Time / s"].iloc[0] / \
            df["Threadscaled Sim. Time / s"]

        # Plot
        sp_up_ax.plot(
            df["NGPUs"].values, df["Speedup"].values,
            marker='*', lw=1.0, label=name)
    sp_up_ax.plot(df["NGPUs"].values, df["Speedup - ideal"].values,
                  ls='dashed', lw=1.0, c='k', label="ideal")
    sp_up_ax.legend(ncol=1)
    sp_up_ax.set_xticks(df["NGPUs"].values)
    sp_up_ax.set_yticks(df["Speedup - ideal"].values)
    sp_up_ax.set_ylabel('Speedup')
    sp_up_ax.set_xlim((0, np.amax(df["NGPUs"].values+1)))
    sp_up_ax.set_ylim((0, np.amax(df["Speedup - ideal"].values+1)))
    sp_up_ax.grid()
    plot_png = f"scaling_plot_{plot_title}.png"
    plt.savefig(plot_png)

    if archive is not None:
        tmp_d = archive
        os.makedirs(tmp_d)
        for csvfile in csv_files:
            shutil.copyfile(csvfile, os.path.join(tmp_d,
                                                  os.path.basename(csvfile)))
        shutil.copyfile(plot_png, os.path.join(tmp_d, plot_png))
        avg_times.to_csv(os.path.join(tmp_d, "avg_times.csv"), index=False)
        archive_name = shutil.make_archive(
            base_name=archive,
            format='gztar',
            root_dir=os.path.dirname(archive),
        )
        shutil.rmtree(tmp_d)
        print("Archived logs and plot at: ", archive_name)


@app.command()
def exec_pipeline(
    config: Annotated[Path, typer.Option(
        help="Path to the configuration file of the pipeline to execute."
    )],
    pipe_key: Annotated[str, typer.Option(
        help=("Key in the configuration file identifying "
              "the pipeline object to execute.")
    )] = "pipeline",
    print_config: Annotated[bool, typer.Option(
        help=("Print config to be executed after overrides.")
    )] = False,
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
    if print_config:
        import json
        print()
        print("#="*15 + " Used configuration " + "#="*15)
        print(json.dumps(parser.config, indent=2))
        print("#="*50)
        print()
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
