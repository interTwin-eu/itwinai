"""
Show how to use TensorFlow MultiWorkerMirroredStrategy on itwinai.

with SLURM:
>>> sbatch tfmirrored_slurm.sh

"""
from typing import Any
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from itwinai.tensorflow.distributed import get_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", "-s", type=str,
        choices=['mirrored'],
        default='mirrored'
    )
    parser.add_argument(
        "--batch_size", "-bs", type=int,
        default=64
    )
    parser.add_argument(
        "--shuffle_dataloader",
        action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()
    return args


def tf_rnd_dataset():
    """Dummy TF dataset."""

    x_train = tf.random.normal((60000, 784), dtype='float32')
    x_test = tf.random.normal((10000, 784), dtype='float32')
    y_train = tf.random.uniform((60000,), minval=0, maxval=10, dtype='int32')
    y_test = tf.random.uniform((10000,), minval=0, maxval=10, dtype='int32')

    return x_train, x_test, y_train, y_test


def trainer_entrypoint_fn(
        foo: Any, args: argparse.Namespace, strategy
) -> int:
    """Dummy training function, similar to custom code developed
    by some use case.
    """
    # dataset to be trained
    x_train, x_test, y_train, y_test = tf_rnd_dataset()

    # distribute datasets among mirrored replicas
    dist_x_train = strategy.experimental_distribute_dataset(
        x_train
    )
    dist_x_test = strategy.experimental_distribute_dataset(
        x_test
    )
    dist_y_train = strategy.experimental_distribute_dataset(
        y_train
    )
    dist_y_test = strategy.experimental_distribute_dataset(
        y_test
    )

    # define and compile model within strategy.scope()
    with strategy.scope():
        # Local model
        inputs = keras.Input(shape=(784,), name='img')
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')

        model.compile(loss=keras.losses.SparseCategoricalCrossentropy
                      (from_logits=True),
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=['accuracy']
                      )

    model.fit(dist_x_train, dist_y_train,
              batch_size=args.batch_size,
              epochs=5,
              validation_split=0.2)

    test_scores = model.evaluate(dist_x_test, dist_y_test, verbose=0)

    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    return 123


if __name__ == "__main__":

    args = parse_args()

    # Instantiate Strategy
    if args.strategy == 'mirrored':
        if (len(tf.config.list_physical_devices('GPU')) == 0):
            raise RuntimeError('Resources unavailable')
        strategy, num_replicas = get_strategy()
    else:
        raise NotImplementedError(
            f"Strategy {args.strategy} is not recognized/implemented.")

    # Launch distributed training
    trainer_entrypoint_fn("foobar", args, strategy)
