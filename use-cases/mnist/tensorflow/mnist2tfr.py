import tensorflow as tf
import numpy as np
import argparse
import os

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

def image_label_to_tf_train(image, label):
    image_shape = np.shape(image)
    #define the dictionary -- the structure -- of our single example
    data = {
        #'height': _int64_feature(image_shape[0]),
        #'width': _int64_feature(image_shape[1]),
        'image' : _bytes_feature(serialize_array(image)),
        'label' : _int64_feature(int(label))
    }
    #create an Example, wrapping the single features
    return tf.train.Example(features=tf.train.Features(feature=data))


def write_images_to_tfr_short(images, labels, filename:str="images", folder = ""):
    filename= folder + "/" + filename+".tfrecords"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0

    for index in range(len(images)):
        #get the data we want to write
        current_image = images[index]
        current_label = labels[index]

        out = image_label_to_tf_train(image=current_image, label=current_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count


if __name__ == '__main__':
    """
    Example usage: python mnist2tfr.py --directory=/media/linx/Secondary/C_T6/use-cases/mnist/tensorflow/data
    """
    parser = argparse.ArgumentParser(
        description="Manage MLFLow server."
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        help="Where to store dataset",
        default=None
    )
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    write_images_to_tfr_short(x_train, y_train, filename= "train", folder = os.path.join(args.directory, 'train'))
    write_images_to_tfr_short(x_test, y_test, filename= "test", folder = os.path.join(args.directory, 'test'))
