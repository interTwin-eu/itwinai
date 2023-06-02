"""
Command line interface for out Python application.
You can call commands from the command line.
Example

>>> $ itwinai --help

"""

# NOTE: import libs in the command"s function, not here.
# Otherwise this will slow the whole CLI.

# from typing import Optional
import os
import sys
import typer


app = typer.Typer()


@app.command()
def train(
    train_dataset: str = typer.Option(
        "unk",
        help="Path to training dataset."
    ),
    config: str = typer.Option(
        "unk",
        help="Path to training configuration file."
    ),
    ml_logs: str = typer.Option(
        "ml-logs/",
        help="Path to logs storage."
    )
):
    """
    Train a neural network defined as a Pytorch Lightning model.
    """
    import copy
    import shutil
    import mlflow
    from lightning.pytorch.cli import LightningCLI
    from omegaconf import DictConfig, OmegaConf

    from itwinai.utils import (
        load_yaml_with_deps,
        check_server,
        flatten_dict
    )
    from itwinai.plmodels.base import (
        ItwinaiBasePlModule,
        ItwinaiBasePlDataModule
    )

    # Replicate args under cli field, to be used in imported config files
    cli_conf = dict(cli=dict(
        train_dataset=train_dataset,
        config=config,
        ml_logs=ml_logs
    ))
    cli_conf = OmegaConf.create(cli_conf)

    # os.makedirs(ml_logs, exist_ok=True)
    train_config: DictConfig = load_yaml_with_deps(config)
    train_config = OmegaConf.merge(train_config, cli_conf)
    # print(OmegaConf.to_yaml(train_config))
    train_config = OmegaConf.to_container(train_config, resolve=True)

    # Setup logger
    if os.path.exists('checkpoints'):
        # Remove old checkpoints
        shutil.rmtree('checkpoints')

    # Check if MLflow server is reachable
    if not check_server(ml_logs):
        raise RuntimeError("MLFlow server not reachable!")

    log_conf = train_config['logger']
    # mlflow.set_tracking_uri('file:' + ml_logs)
    mlflow.set_tracking_uri(ml_logs)
    mlflow.set_experiment(log_conf['experiment_name'])
    mlflow.pytorch.autolog(
        log_every_n_epoch=log_conf['log_every_n_epoch'],
        log_every_n_step=log_conf['log_every_n_steps'],
        registered_model_name=log_conf['registered_model_name']
    )
    # Note: we use autolog and MlFlowLogger combined:
    # - MlFlow logger provides better flexibility
    # - autolog takes care of repetitive operations
    # Ref: https://github.com/Lightning-AI/lightning/discussions/11197

    # Load training configuration
    lightning_conf = train_config['train']['conf']
    # lightning_conf = OmegaConf.to_container(lightning_conf, resolve=True)

    # Start Mlflow run
    with mlflow.start_run(description=log_conf['description']):
        # Log hyperparameters
        config_params = copy.copy(train_config)
        config_params['cli.train_dataset'] = train_dataset
        config_params['cli.ml_logs'] = ml_logs
        config_params['cli.config'] = config
        mlflow.log_params(flatten_dict(config_params))

        # Save config file used for this specific training run
        # for reproducibility
        mlflow.log_artifact(config)

        # Update lightning MLFlow logger constructor args
        # Infer MlFlow conf from pre-configured mlflow client
        lightning_conf['trainer']['logger']['init_args'].update(dict(
            experiment_name=mlflow.get_experiment(
                mlflow.active_run().info.experiment_id
            ).name,
            tracking_uri=mlflow.get_tracking_uri(),
            log_model='all',
            run_id=mlflow.active_run().info.run_id,
            save_dir=None
        ))
        # Append CSVLogger in front:
        # https://github.com/Lightning-AI/lightning/issues/16310#issuecomment-1404177131
        csv_log_conf = dict(
            class_path='lightning.pytorch.loggers.CSVLogger',
            init_args=dict(save_dir='./.tmp')
        )
        lightning_conf['trainer']['logger'] = [
            csv_log_conf,
            lightning_conf['trainer']['logger']
        ]

        # Reset argv before using Lightning CLI
        old_argv = sys.argv
        sys.argv = ['some_script_placeholder.py']
        cli = LightningCLI(
            args=lightning_conf,
            model_class=ItwinaiBasePlModule,
            datamodule_class=ItwinaiBasePlDataModule,
            run=False,
            save_config_kwargs={"overwrite": True,
                                "config_filename": "pl-training.yml"},
            subclass_mode_model=True,
            subclass_mode_data=True
        )
        print(cli.trainer.log_dir)
        sys.argv = old_argv
        # Train + validation, and test
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)
        cli.trainer.test(
            dataloaders=cli.datamodule,
            datamodule=cli.datamodule,
            ckpt_path='best'
        )

        # Save updated lightning conf as an mlflow artifact
        mlflow.log_artifact(
            os.path.join(cli.trainer.log_dir, "pl-training.yml")
        )


@app.command()
def predict(
    input_dataset: str = typer.Option(
        "unk",
        help="Path to dataset of unseen data to make predictions."
    ),
    config: str = typer.Option(
        "unk",
        help="Path to inference configuration file."
    ),
    predictions_location: str = typer.Option(
        "preds/",
        help="Where to save predictions."
    ),
    ml_logs: str = typer.Option(
        "ml-logs/",
        help="Path to MLFLow logs."
    ),
):
    """
    Apply a pre-trained neural network to a set of unseen data.
    """
    import logging
    import mlflow
    from mlflow.exceptions import MlflowException
    from lightning.pytorch.cli import LightningCLI
    from lightning.pytorch.trainer.trainer import Trainer
    import torch
    from omegaconf import DictConfig, OmegaConf

    from itwinai.utils import load_yaml_with_deps, load_yaml
    from itwinai.plmodels.base import (
        ItwinaiBasePlModule,
        ItwinaiBasePlDataModule
    )

    # Replicate args under cli field, to be used in imported config files
    cli_conf = dict(cli=dict(
        input_dataset=input_dataset,
        config=config,
        predictions_location=predictions_location,
        ml_logs=ml_logs
    ))
    cli_conf = OmegaConf.create(cli_conf)

    os.makedirs(predictions_location, exist_ok=True)
    ml_conf: DictConfig = load_yaml_with_deps(config)
    ml_conf = OmegaConf.merge(ml_conf, cli_conf)
    # print(OmegaConf.to_yaml(ml_conf))
    ml_conf = OmegaConf.to_container(ml_conf, resolve=True)
    ml_conf = ml_conf['inference']

    os.makedirs(predictions_location, exist_ok=True)

    # mlflow.set_tracking_uri('file:' + ml_logs)
    mlflow.set_tracking_uri(ml_logs)

    # Check if run ID exists
    try:
        mlflow.get_run(ml_conf['run_id'])
        # mlflow_client.get_run(ml_conf['run_id'])
    except MlflowException:
        logging.warning(
            f"Run ID '{ml_conf['run_id']}' not found! "
            "Using latest run available for experiment "
            f"'{ml_conf['experiment_name']}' instead."
        )
        runs = mlflow.search_runs(
            experiment_names=[ml_conf['experiment_name']],

        )
        new_run_id = runs[runs.status == 'FINISHED'].iloc[0]['run_id']
        ml_conf['run_id'] = new_run_id
        logging.warning(f"Using Run ID: '{new_run_id}'")

    # Download training configuration
    train_conf_path = mlflow.artifacts.download_artifacts(
        run_id=ml_conf['run_id'],
        artifact_path=ml_conf['train_config_artifact_path'],
        dst_path='tmp/',
        tracking_uri=mlflow.get_tracking_uri()
    )

    # Download last ckpt
    ckpt_path = mlflow.artifacts.download_artifacts(
        run_id=ml_conf['run_id'],
        artifact_path=ml_conf['ckpt_path'],
        dst_path='tmp/',
        tracking_uri=mlflow.get_tracking_uri()
    )

    # Instantiate PL model
    lightning_conf = load_yaml(train_conf_path)
    if ml_conf['conf'] is not None:
        # Override training configuration with the one
        # provided during inference.
        # Example: predictions dataset is different
        # from training dataset
        lightning_conf.update(ml_conf['conf'])

    # Reset argv before using Lightning CLI
    old_argv = sys.argv
    sys.argv = ['some_script_placeholder.py']
    cli = LightningCLI(
        args=lightning_conf,
        model_class=ItwinaiBasePlModule,
        run=False,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None
    )
    sys.argv = old_argv

    # Load best model
    loaded_model = cli.model.load_from_checkpoint(
        ckpt_path,
        lightning_conf['model']['init_args']
    )

    # Load Data module
    loaded_data_module: ItwinaiBasePlDataModule = cli.datamodule

    trainer = Trainer(logger=cli.trainer.logger)

    # Predict
    predictions = trainer.predict(
        loaded_model,
        datamodule=loaded_data_module
    )  # , ckpt_path='best')
    pred_class_names = loaded_data_module.preds_to_names(
        torch.cat(predictions)
    )

    # Save list of predictions as class names
    preds_file = os.path.join(predictions_location, 'predictions.txt')
    with open(preds_file, 'w') as preds_file:
        preds_file.write('\n'.join(pred_class_names))


@app.command()
def mlflow_ui(
    path: str = typer.Option(
        "ml-logs/",
        help="Path to logs storage."
    ),
):
    """
    Visualize Mlflow logs.
    """
    import subprocess
    subprocess.run(f"mlflow ui --backend-store-uri {path}".split())


@app.command()
def datasets(
    use_case: str = typer.Option(
        "./use-cases/mnist",
        help="Path to use case files."
    ),
):
    """List datasets of an use case."""
    from rich.console import Console
    from rich.table import Table
    from itwinai.utils import load_yaml
    from pathlib import Path
    from omegaconf import OmegaConf

    datasets_reg = load_yaml(
        os.path.join(use_case, 'meta.yml')
    )
    # datasets_reg = OmegaConf.create(datasets_reg)
    datasets_reg = OmegaConf.to_container(
        OmegaConf.create(datasets_reg),
        resolve=True
    )

    rows = []
    columns = [
        "Name",
        "Description",
        "Location"
    ]
    for ds_name, ds_info in datasets_reg['datasets'].items():
        rows.append(
            [
                ds_name,
                ds_info['doc'],
                ds_info['location']
            ]
        )

    use_case_dir = os.path.basename(Path(use_case))
    table = Table(title=f"Datasets registry for '{use_case_dir}' use case")
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*row, style='bright_green')
    console = Console()
    console.print(table)


@app.command()
def workflows(
    use_case: str = typer.Option(
        "./use-cases/mnist",
        help="Path to use case files."
    ),
):
    """List workflows of an use case."""
    from omegaconf import OmegaConf
    from rich.console import Console
    from rich.table import Table
    from pathlib import Path
    from itwinai.utils import load_yaml_with_deps
    use_case_dir = os.path.basename(Path(use_case))
    wf_files = filter(lambda itm: itm.endswith(
        "-workflow.yml"), os.listdir(use_case))
    columns = [
        "Step",
        "Description",
        "Command",
        "Env location",
        "Env file"
    ]
    for workflow_file in wf_files:
        wf = load_yaml_with_deps(
            os.path.join(use_case, workflow_file)
        )
        wf_name = workflow_file.split('.')[0]
        rows = []
        for step in wf.steps:
            step = OmegaConf.to_container(step, resolve=True)
            step_name, step_data = list(step.items())[0]
            rows.append([
                step_name,
                step_data['doc'],
                step_data['command'],
                step_data['env']['prefix'],
                step_data['env']['file'],
            ])

        table = Table(title=f"'{wf_name}' for '{use_case_dir}' use case")
        for column in columns:
            table.add_column(column)
        for row in rows:
            table.add_row(*row, style='bright_green')
        console = Console()
        console.print(table)


if __name__ == "__main__":
    app()
