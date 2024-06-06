"""
Show how to use TensorFlow MultiWorkerMirroredStrategy on itwinai.

with SLURM:
>>> sbatch tfmirrored_slurm.sh

"""
from typing import Any
import argparse
import tensorflow as tf
from tensorflow import keras
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


def tf_rnd_dataset(args):
    """Dummy TF dataset."""
    (x_train, y_train), (x_test, y_test) = \
        tf.keras.datasets.mnist.load_data(
            path='p/scratch/intertwin/datasets/.keras/datasets/mnist.npz')

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(args.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(args.batch_size)

    return train_dataset, test_dataset


def trainer_entrypoint_fn(
        foo: Any, args: argparse.Namespace, strategy
) -> int:
    """Dummy training function, similar to custom code developed
    by some use case.
    """
    # dataset to be trained
    train_dataset, test_dataset = tf_rnd_dataset(args)

    # distribute datasets among mirrored replicas
    dist_train = strategy.experimental_distribute_dataset(
        train_dataset
    )
    dist_test = strategy.experimental_distribute_dataset(
        test_dataset
    )

    # define and compile model within strategy.scope()
    with strategy.scope():
        # Local model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        model.compile(loss=keras.losses.SparseCategoricalCrossentropy
                      (from_logits=True),
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=['accuracy']
                      )

    model.fit(dist_train,
              epochs=5,
              steps_per_epoch=2000)

    test_scores = model.evaluate(dist_test, verbose=0, steps=500)

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
