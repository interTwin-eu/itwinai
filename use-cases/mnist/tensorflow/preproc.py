import tensorflow as tf
import argparse
import os

"""
File for preprocessing MNIST dataset into .tfrecord files in order to be used for TFX pipeline
"""


def convert_bytes(value):
    """
    Convert bytes value into bytes tf feature
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_int64(value):
    """
    Convert int64 value into bytes tf feature
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    """
    Serializes array into tensorflow tensor
    """
    return tf.io.serialize_tensor(array)

def create_example(image, label):
    """
    Creates tf Example based on MNIST features
    """
    data = {
        'image' : convert_bytes(serialize_array(image)),
        'label' : convert_int64(int(label))
    }
    return tf.train.Example(features=tf.train.Features(feature=data))


def create_tfrecord(images, labels, filepath):
    """
    Writes a tf record based on the image, label and filepath given
    """
    print(f'Creating tfrecords at: {filepath}')
    writer = tf.io.TFRecordWriter(filepath)
    for index in range(len(images)):
        current_image, current_label = images[index], labels[index]
        example = create_example(image=current_image, label=current_label)
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a tfrecord for MNIST dataset"
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        help="Where to store dataset (train, test subfolders)",
        default=None
    )
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Create dirs if not exists
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
        os.makedirs(os.path.join(args.directory, 'train'))
        os.makedirs(os.path.join(args.directory, 'test'))

    create_tfrecord(x_train, y_train, filepath = os.path.join(args.directory, 'train/train.tfrecords'))
    create_tfrecord(x_test, y_test, filepath = os.path.join(args.directory, 'test/test.tfrecords'))
