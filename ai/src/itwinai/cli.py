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
import yaml
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
    ),
):
    """
    Train a neural network expressed as a Pytorch Lightning model.
    """
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger
    from itwinai.utils import dynamically_import_class

    with open(config, "r", encoding="utf-8") as yaml_file:
        try:
            train_config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc

    model_class = dynamically_import_class(train_config["model"])
    model = model_class(input, **train_config["hyperparams"])

    os.makedirs(output, exist_ok=True)
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=3,
        logger=CSVLogger(save_dir=output),
    )

    trainer.fit(model)


@app.command()
def hello():
    """Say hello"""
    print("hello")


if __name__ == "__main__":
    app()
