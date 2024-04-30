from typing import Dict, Any
import logging
from os.path import join, exists

import tensorflow.keras as keras

from lib.strategy import get_mirrored_strategy
from lib.utils import get_network_config, load_model
from itwinai.components import Trainer, monitor_exec
from lib.callbacks import ProcessBenchmark
from lib.macros import (
    Network,
    Losses,
    RegularizationStrength,
    Activation,
)


class TensorflowTrainer(Trainer):
    def __init__(
        self,
        network: Network,
        activation: Activation,
        regularization_strength: RegularizationStrength,
        learning_rate: float,
        loss: Losses,
        epochs: int,
        batch_size: int,
        global_config: Dict[str, Any],
        kernel_size: int = None,
        model_backup: str = None,
        cores: int = None,
    ):
        super().__init__()
        self.save_parameters(**self.locals2params(locals()))
        self.epochs = epochs
        self.batch_size = batch_size
        self.global_config = global_config
        self.cores = cores
        self.model_backup = model_backup
        self.network = network.value
        self.activation = activation.value
        self.kernel_size = kernel_size
        self.regularization_strength, self.regularizer = (
            regularization_strength.value
        )

        # Loss name and learning rate
        self.loss_name = loss.value
        self.learning_rate = learning_rate

        # Parse global config
        self.setup_config(self.global_config)

    @monitor_exec
    def execute(self, train_data, validation_data, channels) -> None:
        train_dataset, n_train = train_data
        valid_dataset, n_valid = validation_data

        # set mirrored strategy
        mirrored_strategy, n_devices = get_mirrored_strategy(cores=self.cores)
        logging.debug(f"Mirrored strategy created with {n_devices} devices")

        # distribute datasets among MirroredStrategy's replicas
        dist_train_dataset = mirrored_strategy.experimental_distribute_dataset(
            train_dataset
        )
        dist_valid_dataset = mirrored_strategy.experimental_distribute_dataset(
            valid_dataset
        )

        # Inside the strategy load the model, data generators and train
        with mirrored_strategy.scope():
            if not self.model_backup:
                model = get_network_config(
                    network=self.network,
                    patch_size=self.patch_size,
                    activation=self.activation,
                    regularizer=self.regularizer,
                    kernel_size=self.kernel_size,
                    channels=channels,
                )
                logging.debug("New model created")
            else:
                model = load_model(model_fpath=self.best_model_name)
                logging.debug(
                    f"Model loaded from backup at {self.best_model_name}")

            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
            metrics = [keras.metrics.MeanAbsoluteError(name="mae")]
            model.compile(loss=self.loss_name,
                          optimizer=optimizer, metrics=metrics)
        logging.debug("Model compiled")

        # print model summary to check if model's architecture is correct
        print(model.summary())

        # compute the steps per epoch for train and valid
        steps_per_epoch = n_train // self.batch_size
        validation_steps = n_valid // self.batch_size

        # train the model
        model.fit(
            dist_train_dataset,
            validation_data=dist_valid_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=self.epochs,
            callbacks=self.callbacks,
        )
        logging.debug("Model trained")

        # save the best model
        model.save(self.last_model_name)
        logging.debug("Saved training history")

    def setup_config(self, config: Dict) -> None:
        self.experiment_dir = config["experiment_dir"]
        self.run_dir = config["run_dir"]
        self.patch_size = config["patch_size"]

        # Paths
        CHECKPOINTS_DIR = join(self.run_dir, "checkpoints")

        # files and csvs definition
        CHECKPOINTS_FILEPATH = join(CHECKPOINTS_DIR, "model_{epoch:02d}.h5")
        LOSS_METRICS_HISTORY_CSV = join(
            self.run_dir, "loss_metrics_history.csv")
        BENCHMARK_HISTORY_CSV = join(self.run_dir, "benchmark_history.csv")

        self.callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=100,
                min_delta=0.0001,
                restore_best_weights=True,
                verbose=1,
                mode="min",
            ),
            keras.callbacks.CSVLogger(LOSS_METRICS_HISTORY_CSV),
            ProcessBenchmark(BENCHMARK_HISTORY_CSV),
            keras.callbacks.ModelCheckpoint(
                filepath=CHECKPOINTS_FILEPATH,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                save_weights_only=False,
                verbose=1,
            ),
        ]

        # Check if model backup exists
        if self.model_backup is not None and not exists(self.model_backup):
            raise FileNotFoundError("Model backup file not found")
        if self.model_backup:
            self.best_model_name = join(self.model_backup, "best_model.h5")
        self.last_model_name = join(self.run_dir, "last_model.h5")
