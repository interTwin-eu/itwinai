# Adapted from:
import tensorflow_datasets as tfds
import tensorflow as tf
import keras


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label


if __name__ == '__main__':

    print(tf.__version__)
    datasets, info = tfds.load(
        name='mnist', with_info=True, as_supervised=True)

    mnist_train, mnist_test = datasets['train'], datasets['test']
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # You can also do info.splits.total_num_examples to get the total
    # number of examples in the dataset.

    num_train_examples = info.splits['train'].num_examples
    num_test_examples = info.splits['test'].num_examples

    BUFFER_SIZE = 10000

    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    train_dataset = mnist_train.map(scale).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

    with strategy.scope():
        # # This model is broken!
        # model = tf.keras.Sequential(
        #     [
        #         tf.keras.Input(shape=(28, 28, 1)),
        #         tf.keras.layers.Conv2D(
        #             32, kernel_size=3, activation="relu"),
        #         tf.keras.layers.MaxPooling2D(),
        #         tf.keras.layers.Conv2D(
        #             64, kernel_size=3, activation="relu"),
        #         tf.keras.layers.MaxPooling2D(pool_size=2),
        #         tf.keras.layers.Flatten(),
        #         tf.keras.layers.Dropout(0.5),
        #         tf.keras.layers.Dense(10)
        #     ]
        # )

        # # This model (from the example) works
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(
        #         32, 3, activation='relu', input_shape=(28, 28, 1)),
        #     tf.keras.layers.MaxPooling2D(),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(64, activation='relu'),
        #     tf.keras.layers.Dense(10)
        # ])

        model = keras.Sequential([

            keras.layers.Conv2D(filters=6, kernel_size=(
                3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.AveragePooling2D(),
            keras.layers.Conv2D(
                filters=16, kernel_size=(3, 3), activation='relu'),
            keras.layers.AveragePooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(units=120, activation='relu'),
            keras.layers.Dense(units=84, activation='relu'),
            keras.layers.Dense(units=10)
        ])

        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])

    callbacks = []

    EPOCHS = 12

    model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)
