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
import typer


app = typer.Typer()


@app.command()
def train(
    input: str = typer.Option(
        "unk",
        help="Path to training dataset."
    ),
    config: str = typer.Option(
        "unk",
        help="Path to training configuration file."
    ),
    output: str = typer.Option(
        "logs/",
        help="Path to logs storage."
    )
):
    """
    Train a neural network defined as a Pytorch Lightning model.
    """
    import copy
    import mlflow
    from lightning.pytorch.cli import LightningCLI
    from omegaconf import DictConfig, OmegaConf

    from itwinai.utils import load_yaml_with_deps, flatten_dict
    from itwinai.plmodels.base import (
        ItwinaiBasePlModule,
        ItwinaiBasePlDataModule
    )
    cli_conf = dict(cli=dict(
        input_dataset=input,
        ml_logs=output
    ))
    cli_conf = OmegaConf.create(cli_conf)

    os.makedirs(output, exist_ok=True)
    train_config: DictConfig = load_yaml_with_deps(config)
    train_config = OmegaConf.merge(train_config, cli_conf)
    # print(OmegaConf.to_yaml(train_config))
    train_config = OmegaConf.to_container(train_config, resolve=True)

    log_conf = train_config['logger']
    mlflow.set_tracking_uri('file:' + output)
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
        config_params['cli.input'] = input
        config_params['cli.output'] = output
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
            run_id=mlflow.active_run().info.run_id
        ))

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

        # Train + validation, and test
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)
        cli.trainer.test(dataloaders=cli.datamodule, datamodule=cli.datamodule)

        # Save updated lightning conf as an mlflow artifact
        mlflow.log_artifact(
            os.path.join(cli.trainer.log_dir, "pl-training.yml")
        )


@app.command()
def predict(
    input: str = typer.Option(
        "unk",
        help="Path to predictions dataset."
    ),
    config: str = typer.Option(
        "unk",
        help="Path to inference configuration file."
    ),
    output: str = typer.Option(
        "preds/",
        help="Path to predictions storage."
    ),
    ml_logs: str = typer.Option(
        "logs/",
        help="Path to MLFLow logs."
    ),
):
    """
    Apply a pre-trained neural network to a set of unseen data.
    """
    import mlflow
    from lightning.pytorch.cli import LightningCLI
    from lightning.pytorch.trainer.trainer import Trainer
    import torch
    from omegaconf import DictConfig, OmegaConf

    from itwinai.utils import load_yaml_with_deps, load_yaml
    from itwinai.plmodels.base import (
        ItwinaiBasePlModule,
        ItwinaiBasePlDataModule
    )

    # TODO: define input as PL dataModule

    os.makedirs(output, exist_ok=True)
    ml_conf: DictConfig = load_yaml_with_deps(config)
    # print(OmegaConf.to_yaml(ml_conf))
    ml_conf = OmegaConf.to_container(ml_conf, resolve=True)
    ml_conf = ml_conf['inference']

    os.makedirs(output, exist_ok=True)

    mlflow.set_tracking_uri('file:' + ml_logs)

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

    cli = LightningCLI(
        args=lightning_conf,
        model_class=ItwinaiBasePlModule,
        run=False,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None
    )

    # Load best model
    loaded_model = cli.model.load_from_checkpoint(
        ckpt_path,
        lightning_conf['model']['init_args']
    )

    # Load Data module
    if ml_conf.get('data') is not None:
        # New/updated datamodule
        loaded_data_module: ItwinaiBasePlDataModule = None
        raise NotImplementedError
    else:
        # Reuse same datamodule used for training
        loaded_data_module: ItwinaiBasePlDataModule = cli.datamodule

    # Test best model once again (TODO: remove)
    trainer = Trainer()
    trainer.test(
        loaded_model,
        dataloaders=loaded_data_module,
        datamodule=loaded_data_module
    )  # , ckpt_path='best')

    # Predict
    predictions = trainer.predict(
        loaded_model,
        datamodule=loaded_data_module
    )  # , ckpt_path='best')
    pred_class_names = loaded_data_module.preds_to_names(
        torch.cat(predictions)
    )

    # Save list of predictions as class names
    with open(os.path.join(output, 'predictions.txt'), 'w') as preds_file:
        preds_file.write('\n'.join(pred_class_names))


@app.command()
def visualize(
    path: str = typer.Option(
        "logs/",
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
    datasets_reg = load_yaml(
        os.path.join(use_case, 'datasets-registry.yml')
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

    use_case_dir = os.path.basename(use_case.strip('/'))
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
    # TODO


if __name__ == "__main__":
    app()
