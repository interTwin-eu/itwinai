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
    Train a neural network expressed as a Pytorch Lightning model.
    """
    import copy
    import yaml
    import lightning as L
    from lightning.pytorch.loggers import MLFlowLogger
    import mlflow

    from itwinai.utils import dynamically_import_class, flatten_dict

    os.makedirs(output, exist_ok=True)

    with open(config, "r", encoding="utf-8") as yaml_file:
        try:
            train_config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc

    model_class = dynamically_import_class(train_config["model"])
    model = model_class(input, **train_config["hyperparams"])

    mlflow.set_tracking_uri("file:" + output)
    mlflow.set_experiment(train_config['experiment_name'])
    mlflow.pytorch.autolog(log_every_n_step=train_config['log_every_n_steps'])

    # Note: we use autolog and MlFlowLogger combined:
    # - MlFlow logger provides better flexibility
    # - autolog takes care of repetitive operations
    # Ref: https://github.com/Lightning-AI/lightning/discussions/11197

    # Start Mlflow run
    with mlflow.start_run(description=train_config['description']):
        # Log hyperparameters
        config_params = copy.copy(train_config)
        config_params['input'] = input
        config_params['output'] = output
        config_params['config'] = config
        mlflow.log_params(flatten_dict(config_params['hyperparams']))
        del config_params['hyperparams']
        mlflow.log_params(flatten_dict(config_params))
        # Save config file used for this specific training run
        # for reproducibility
        mlflow.log_artifact(config)

        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow.get_experiment(
                mlflow.active_run().info.experiment_id
            ).name,
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=mlflow.active_run().info.run_id,
            log_model='all'
        )

        trainer = L.Trainer(
            accelerator=train_config["accelerator"],
            devices=train_config["devices"],
            max_epochs=train_config["max_epochs"],
            logger=mlflow_logger
        )

        # Train + validation
        trainer.fit(model)
        # Test
        trainer.test()


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


if __name__ == "__main__":
    app()
