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
        "--data_dir", type=str,
        default='/p/scratch/intertwin/datasets/.keras/datasets/mnist.npz'
    )
    parser.add_argument(
        "--shuffle_dataloader",
        action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()
    return args

def trainer_entrypoint_fn(
        foo: Any, 
        args: argparse.Namespace, 
        strategy,
        num_replicas
) -> int:
    """Training function, similar to custom code developed
    by some use case.
    """
    # dataset to be trained
    (x_train, y_train), (x_test, y_test) = \
        tf.keras.datasets.mnist.load_data(
            path=args.data_dir)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # scale batch size with number of workers
    batch_size = args.batch_size * num_replicas

    # batching dataset and repeat
    train_dataset = train_dataset.batch(batch_size).repeat()
    test_dataset = test_dataset.batch(batch_size).repeat()
    
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
                  steps_per_epoch=len(x_train)//batch_size)

        test_scores = model.evaluate(dist_test, 
                                     verbose=1, 
                                     steps=len(x_test)//batch_size)

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
    trainer_entrypoint_fn("foobar", args, strategy, num_replicas)
