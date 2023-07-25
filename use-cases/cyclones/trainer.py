import tensorflow.keras as keras
import logging
import tensorflow as tf

from datetime import datetime
from tqdm import tqdm
from os.path import join, exists
from lib.strategy import get_mirrored_strategy
from lib.models.setup import get_network_config, load_model
from lib.callbacks import ProcessBenchmark
from lib.macros import (
    PatchType,
    Network,
    Losses,
    RegularizationStrength,
    Activation,
    LabelNoCyclone,
    AugmentationType,
)

# TODO: Abstraction for Trainer
class TensorflowTrainer:
    def __init__(self, strategy, loss, epochs, batch_size, callbacks, optimizer, model_func, metrics_func):
        self.strategy = strategy
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.callbacks = callbacks
        self.global_batch_size = self.batch_size * self.strategy.num_replicas_in_sync
        self.optimizer = optimizer

        # Create distributed TF vars
        with self.strategy.scope():
            self.model = model_func()
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=metrics_func())

    def train(self, data):
        (train, n_train), (test, n_test) = data
        train = self.strategy.experimental_distribute_dataset(train)
        test = self.strategy.experimental_distribute_dataset(test)

        # compute the steps per epoch for train and valid
        train_steps = n_train // self.batch_size
        test_steps = n_test // self.batch_size

        # train the model
        self.model.fit(
            train,
            validation_data=test,
            steps_per_epoch=train_steps,
            validation_steps=test_steps,
            epochs=self.epochs,
            callbacks=self.callbacks,
        )

        logging.debug(f"Model trained")

class CyclonesTrainer(TensorflowTrainer):
    def __init__(
        self,
        RUN_DIR,
        epochs,
        network,
        activation,
        regularization_strength,
        learning_rate: float,
        loss,
        channels,
        batch_size,
        patch_size,
        kernel_size: int = None,
    ):
        # Configurable
        regularization_strength, regularizer = \
        [rg.value for rg in RegularizationStrength if rg.name.lower() == regularization_strength][0]
        loss_name, loss = [l.value for l in Losses if l.name.lower() == loss][0]
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Paths, Folders
        SCALER_DIR = join(RUN_DIR, "scalers")
        TENSORBOARD_DIR = join(RUN_DIR, "tensorboard")
        CHECKPOINTS_DIR = join(RUN_DIR, "checkpoints")

        # files and csvs definition
        CHECKPOINTS_FILEPATH = join(CHECKPOINTS_DIR, "model_{epoch:02d}.h5")
        LOSS_METRICS_HISTORY_CSV = join(RUN_DIR, "loss_metrics_history.csv")
        BENCHMARK_HISTORY_CSV = join(RUN_DIR, "benchmark_history.csv")

        super().__init__(
            strategy=tf.distribute.MirroredStrategy(),
            loss=loss,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
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
            ],
            optimizer=optimizer,
            model_func=lambda: get_network_config(
                network=network,
                patch_size=patch_size,
                activation=activation,
                regularizer=regularizer,
                kernel_size=kernel_size,
                channels=channels,
            ),
            metrics_func=lambda: [keras.metrics.MeanAbsoluteError(name="mae")]
        )

    def train(self, data):
        # TODO: Integrate train_step locally and pass it into the super distributed training function
        super().train(data)

