# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Roman Machacek
#
# Credit:
# - Roman Machacek <roman.machacek@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import logging
from os.path import exists, join
from typing import Any, Dict, Optional, Union

import tensorflow as tf
import tensorflow.keras as keras

from itwinai.components import monitor_exec
from itwinai.tensorflow.trainer import TensorflowTrainer
from src.callbacks import ProcessBenchmark
from src.macros import Activation, Losses, Network, RegularizationStrength
from src.utils import get_network_config, load_model


class CyclonesTrainer(TensorflowTrainer):
    strategy: tf.distribute.Strategy
    num_workers: int

    def __init__(
        self,
        network: Network,
        activation: Activation,
        regularization_strength: RegularizationStrength,
        learning_rate: float,
        loss: Losses,
        epochs: int,
        micro_batch_size: int,
        global_config: Dict[str, Any],
        kernel_size: Optional[int] = None,
        model_backup: Optional[str] = None,
        rnd_seed: Optional[int] = None,
        verbose: Union[str, int] = "auto",
    ):
        super().__init__(
            epochs=epochs,
            micro_batch_size=micro_batch_size,
            rnd_seed=rnd_seed,
            verbose=verbose,
        )
        self.save_parameters(**self.locals2params(locals()))
        self.global_config = global_config
        self.model_backup = model_backup
        self.network = network.value
        self.activation = activation.value
        self.kernel_size = kernel_size
        self.regularization_strength, self.regularizer = regularization_strength.value

        # Loss name and learning rate
        self.loss_name, self.loss = loss.value
        self.learning_rate = learning_rate

        # Parse global config
        self.dynamic_config(self.global_config)

    @monitor_exec
    def execute(self, train_data, validation_data, channels) -> None:
        # train_size and valid_size are the number of unique elements
        # in the dataset, before calling tf.data.Dataset.repeat(num_epochs)
        train_dataset, train_size = train_data
        valid_dataset, valid_size = validation_data

        # Batch and distribute datasets among strategy's replica.
        # Each batch is further split among the workers
        dist_train_dataset = self.strategy.experimental_distribute_dataset(
            train_dataset.batch(
                self.macro_batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
            )
        )
        dist_valid_dataset = self.strategy.experimental_distribute_dataset(
            valid_dataset.batch(
                self.macro_batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
            )
        )

        # Inside the strategy load the model, data generators and train
        with self.strategy.scope():
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
                logging.debug(f"Model loaded from backup at {self.best_model_name}")

            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
            metrics = [keras.metrics.MeanAbsoluteError(name="mae")]
            model.compile(loss=self.loss_name, optimizer=optimizer, metrics=metrics)
        logging.debug("Model compiled")

        # Print model summary to check if model's architecture is correct
        print(model.summary())

        # Compute the steps per epoch for train and valid
        steps_per_epoch = train_size // self.macro_batch_size
        validation_steps = valid_size // self.macro_batch_size

        print("macro_batch_size: ", self.macro_batch_size, flush=True)

        # Train the model
        model.fit(
            dist_train_dataset,
            validation_data=dist_valid_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=self.epochs,
            callbacks=self.callbacks,
        )
        logging.debug("Model trained")

        # Save the best model
        model.save(self.last_model_name)
        logging.debug("Saved training history")

    def dynamic_config(self, config: Dict) -> None:
        """Parse configuration generated at runtime."""
        self.experiment_dir = config["experiment_dir"]
        self.run_dir = config["run_dir"]
        self.patch_size = config["patch_size"]

        # Paths
        CHECKPOINTS_DIR = join(self.run_dir, "checkpoints")

        # files and csvs definition
        CHECKPOINTS_FILEPATH = join(CHECKPOINTS_DIR, "model_{epoch:02d}.keras")
        LOSS_METRICS_HISTORY_CSV = join(self.run_dir, "loss_metrics_history.csv")
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
            self.best_model_name = join(self.model_backup, "best_model.keras")
        self.last_model_name = join(self.run_dir, "last_model.keras")
