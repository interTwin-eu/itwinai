import tensorflow.keras as keras
import tensorflow as tf

from os.path import join
from lib.models.setup import get_network_config
from lib.callbacks import ProcessBenchmark
from lib.macros import (
    Losses,
    RegularizationStrength,
)
from itwinai.backend.tensorflow.trainer import TensorflowTrainer


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
        CHECKPOINTS_DIR = join(RUN_DIR, "checkpoints")

        # files and csvs definition
        CHECKPOINTS_FILEPATH = join(CHECKPOINTS_DIR, "model_{epoch:02d}.h5")
        LOSS_METRICS_HISTORY_CSV = join(RUN_DIR, "loss_metrics_history.csv")
        BENCHMARK_HISTORY_CSV = join(RUN_DIR, "benchmark_history.csv")

        super().__init__(
            strategy=None,  # tf.distribute.MirroredStrategy(),
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
            metrics_func=lambda: [keras.metrics.MeanAbsoluteError(name="mae")],
        )

    def train(self, data):
        super().train(data)
