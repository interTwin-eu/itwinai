# CLI

**Usage**:

```console
$ [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `generate-scalability-report`: Generates scalability reports for epoch...
* `sanity-check`: Run sanity checks on the installation of...
* `exec-pipeline`: Execute a pipeline from configuration file.
* `mlflow-ui`: Visualize Mlflow logs.
* `mlflow-server`: Spawn Mlflow server.
* `kill-mlflow-server`: Kill Mlflow server.

## `generate-scalability-report`

Generates scalability reports for epoch time, GPU data, and communication data
based on log files in the specified directory. Optionally, backups of the reports
can be created.

This command processes log files stored in specific subdirectories under the given
`log_dir`. It generates plots and metrics for scalability analysis and saves them
in the `plot_dir`. If backups are enabled, the generated reports will also be
copied to a backup directory under `backup_root_dir`.

**Usage**:

```console
$ generate-scalability-report [OPTIONS]
```

**Options**:

* `--log-dir TEXT`: Which directory to search for the scalability metrics in.  [default: scalability-metrics]
* `--plot-dir TEXT`: Which directory to save the resulting plots in.  [default: plots]
* `--do-backup / --no-do-backup`: Whether to store a backup of the scalability metrics that were used to make the report or not.  [default: no-do-backup]
* `--backup-root-dir TEXT`: Which directory to store the backup files in.  [default: backup-scalability-metrics/]
* `--experiment-name TEXT`: What to name the experiment in the backup directory. Will be automatically generated if left as None.
* `--help`: Show this message and exit.

## `sanity-check`

Run sanity checks on the installation of itwinai and its dependencies by trying
to import itwinai modules. By default, only itwinai core modules (neither torch, nor
tensorflow) are tested.

**Usage**:

```console
$ sanity-check [OPTIONS]
```

**Options**:

* `--torch / --no-torch`: Check also itwinai.torch modules.  [default: no-torch]
* `--tensorflow / --no-tensorflow`: Check also itwinai.tensorflow modules.  [default: no-tensorflow]
* `--all / --no-all`: Check all modules.  [default: no-all]
* `--optional-deps TEXT`: List of optional dependencies.
* `--help`: Show this message and exit.

## `exec-pipeline`

Execute a pipeline from configuration file. Allows dynamic override of fields.

**Usage**:

```console
$ exec-pipeline [OPTIONS]
```

**Options**:

* `--config PATH`: Path to the configuration file of the pipeline to execute.  [required]
* `--pipe-key TEXT`: Key in the configuration file identifying the pipeline object to execute.  [default: pipeline]
* `--steps TEXT`: Run only some steps of the pipeline. Accepted values are indices, python slices (e.g., 0:3 or 2:10:100), and string names of steps.
* `--print-config / --no-print-config`: Print config to be executed after overrides.  [default: no-print-config]
* `-o, --override TEXT`: Nested key to dynamically override elements in the configuration file with the corresponding new value, joined by &#x27;=&#x27;. It is also possible to index elements in lists using their list index. Example: [...] -o pipeline.init_args.trainer.init_args.lr=0.001 -o pipeline.my_list.2.batch_size=64
* `--help`: Show this message and exit.

## `mlflow-ui`

Visualize Mlflow logs.

**Usage**:

```console
$ mlflow-ui [OPTIONS]
```

**Options**:

* `--path TEXT`: Path to logs storage.  [default: ml-logs/]
* `--port INTEGER`: Port on which the MLFlow UI is listening.  [default: 5000]
* `--help`: Show this message and exit.

## `mlflow-server`

Spawn Mlflow server.

**Usage**:

```console
$ mlflow-server [OPTIONS]
```

**Options**:

* `--path TEXT`: Path to logs storage.  [default: ml-logs/]
* `--port INTEGER`: Port on which the server is listening.  [default: 5000]
* `--help`: Show this message and exit.

## `kill-mlflow-server`

Kill Mlflow server.

**Usage**:

```console
$ kill-mlflow-server [OPTIONS]
```

**Options**:

* `--port INTEGER`: Port on which the server is listening.  [default: 5000]
* `--help`: Show this message and exit.
