from typing import Tuple, Dict, Optional

# import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_datasets as tfds

from itwinai.components import DataGetter


class Zebra2HorseDataLoader(DataGetter):
    def __init__(self, buffer_size: int):
        super().__init__()
        self.buffer_size = buffer_size

    def load(self):
        # Load the horse-zebra dataset using tensorflow-datasets.
        dataset, _ = tfds.load("cycle_gan/horse2zebra",
                               with_info=True, as_supervised=True)
        train_horses, train_zebras = dataset["trainA"], dataset["trainB"]
        test_horses, test_zebras = dataset["testA"], dataset["testB"]

        # Image sizes
        orig_img_size = (286, 286)
        input_img_size = (256, 256, 3)

        def normalize_img(img):
            img = tf.cast(img, dtype=tf.float32)
            # Map values in the range [-1, 1]
            return (img / 127.5) - 1.0

        def preproc_train_fn(img, label):
            # Random flip
            img = tf.image.random_flip_left_right(img)
            # Resize to the original size first
            img = tf.image.resize(img, [*orig_img_size])
            # Random crop to 256X256
            img = tf.image.random_crop(img, size=[*input_img_size])
            # Normalize the pixel values in the range [-1, 1]
            img = normalize_img(img)
            return img

        def preproc_test_fn(img, label):
            # Only resizing and normalization for the test images.
            img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
            img = normalize_img(img)
            return img

        # TODO: Add shuffle?
        # Apply the preprocessing operations to the training data
        train_horses = (
            train_horses.map(preproc_train_fn,
                             num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
        )
        train_zebras = (
            train_zebras.map(preproc_train_fn,
                             num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
        )

        # Apply the preprocessing operations to the test data
        test_horses = (
            test_horses.map(preproc_test_fn,
                            num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
        )
        test_zebras = (
            test_zebras.map(preproc_test_fn,
                            num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
        )

        return (
            tf.data.Dataset.zip((train_horses, train_zebras)
                                ).shuffle(self.buffer_size),
            tf.data.Dataset.zip((test_horses, test_zebras)
                                ).shuffle(self.buffer_size)
        )

    def execute(
        self,
        config: Optional[Dict] = None
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        train, test = self.load()
        return ([train, test],), config
