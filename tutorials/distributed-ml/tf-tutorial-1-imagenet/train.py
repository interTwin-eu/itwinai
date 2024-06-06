"""
 Show how to use TensorFlow MultiWorkerMirroredStrategy on itwinai.
 for an Imagenet dataset
 with SLURM:
 >>> sbatch tfmirrored_slurm.sh

 """
import argparse
import sys
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from itwinai.tensorflow.distributed import get_strategy


def parse_args():
    """
    Parse args
    """
    parser = argparse.ArgumentParser(description='TensorFlow ImageNet')

    parser.add_argument(
        "--strategy", "-s", type=str,
        choices=['mirrored'],
        default='mirrored'
    )
    parser.add_argument(
        "--data_dir", type=str,
        default='./'
    )
    parser.add_argument(
        "--batch_size", type=int,
        default=128
    )
    parser.add_argument(
        "--epochs", type=int,
        default=3
    )

    args = parser.parse_args()
    return args


def deserialization_fn(serialized_fn):
    """Imagenet data processing

    Args:
        serialized_example (Any): Input function

    Returns:
        Any: Images and associated labels
    """
    parsed_example = tf.io.parse_single_example(
        serialized_fn,
        features={
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        }
    )
    image = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=3)
    image = tf.image.resize(image, (224, 224))
    label = tf.cast(parsed_example['image/class/label'], tf.int64) - 1
    return image, label


def tf_records_loader(files_path, shuffle=False):
    """tf_records dataset reader

    Args:
        files_path (String): Path to location of data
        shuffle (bool, optional): If dataset should be shuffled.
        Defaults to False.

    Returns:
        tf.data.Dataset: Returns dataset to be trained
    """
    datasets = tf.data.Dataset.from_tensor_slices(files_path)
    datasets = datasets.shuffle(len(files_path)) if shuffle else datasets
    datasets = datasets.flat_map(tf.data.TFRecordDataset)
    datasets = datasets.map(
        deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return datasets


def main():
    args = parse_args()

    input_shape = (224, 224, 3)
    num_classes = 1000

    if args.strategy == 'mirrored':
        strategy = get_strategy()[0]
    else:
        raise NotImplementedError(
            f"Strategy {args.strategy} is not recognized/implemented.")

    with strategy.scope():
        base_model = keras.applications.ResNet50(
            weights=None,
            input_shape=input_shape,
            include_top=False,
        )

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy']
                      )

    # scale batch size with number of workers
    batch_size = args.batch_size * get_strategy()[1]

    dir_imagenet = args.data_dir+'imagenet-1K-tfrecords'
    train_shard_suffix = 'train-*-of-01024'
    test_shard_suffix = 'validation-*-of-00128'

    train_set_path = sorted(
        tf.io.gfile.glob(dir_imagenet + f'/{train_shard_suffix}')
    )
    test_set_path = sorted(
        tf.io.gfile.glob(dir_imagenet + f'/{test_shard_suffix}')
    )

    train_dataset = tf_records_loader(train_set_path, shuffle=True)
    test_dataset = tf_records_loader(test_set_path)

    train_dataset = train_dataset.batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # distribute datasets among mirrored replicas
    dist_train = strategy.experimental_distribute_dataset(
        train_dataset
    )
    dist_test = strategy.experimental_distribute_dataset(
        test_dataset
    )

    # TODO: add callbacks to evaluate per epoch time
    et = timer()

    # trains the model
    model.fit(dist_train, epochs=args.epochs, steps_per_epoch=500, verbose=10)

    print('TIMER: total epoch time:',
          timer() - et, ' s')
    print('TIMER: average epoch time:',
          (timer() - et) / (args.epochs), ' s')

    test_scores = model.evaluate(dist_test, steps=100, verbose=5)

    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])


if __name__ == "__main__":
    main()
    sys.exit()

# eof
