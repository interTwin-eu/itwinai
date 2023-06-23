import tensorflow as tf
import absl

def MNISTModel(input_name) -> tf.keras.Model:
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=(28, 28), name=input_name))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10))

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.RMSprop(lr=0.0015),
        metrics=['sparse_categorical_accuracy']
    )
    model.summary(print_fn=absl.logging.info)
    return model