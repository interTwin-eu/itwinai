from typing import Dict, Optional, Tuple
import logging
from os.path import join, exists

import tensorflow.keras as keras

from lib.strategy import get_mirrored_strategy
from lib.utils import get_network_config, load_model
from itwinai.backend.components import Trainer
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
        kernel_size: int = None,
        model_backup: str = None,
        cores: int = None,
    ):
        super().__init__()
        # Configurable
        self.cores = cores
        self.model_backup = model_backup
        self.network = network.value
        self.activation = activation.value
        self.kernel_size = kernel_size
        self.regularization_strength, self.regularizer = (
            regularization_strength.value
        )
        self.loss_name, self.loss = loss.value

        # Optimizers, Losses
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    def train(self, train_data, validation_data):
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
                    channels=self.channels,
                )
                logging.debug("New model created")
            else:
                model = load_model(model_fpath=self.best_model_name)
                logging.debug(
                    f"Model loaded from backup at {self.best_model_name}")

            metrics = [keras.metrics.MeanAbsoluteError(name="mae")]
            model.compile(loss=self.loss,
                          optimizer=self.optimizer, metrics=metrics)
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

    def execute(
        self,
        train_dataset,
        validation_dataset,
        config: Optional[Dict] = None,
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        config = self.setup_config(config)
        train_result = self.train(train_dataset, validation_dataset)
        return (train_result,), config

    def setup_config(self, config: Optional[Dict] = None) -> Dict:
        config = config if config is not None else {}
        self.experiment_dir = config["experiment_dir"]
        self.run_dir = config["run_dir"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.patch_size = config["patch_size"]
        self.channels = config["channels"]

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

        return config

    def load_state(self):
        return super().load_state()

    def save_state(self):
        return super().save_state()
