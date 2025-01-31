# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: CMCC
#
# Credit:
# - CMCC
# --------------------------------------------------------------------------------------


import tensorflow as tf


def custom_VGG_V1(patch_size, channels, activation, regularizer):
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Conv2D(
            input_shape=(patch_size, patch_size, channels[0]),
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(2, 2),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(2, 2),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(
        tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(2, 2),
            padding="valid",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(2, 2),
            padding="valid",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(2, 2),
            padding="valid",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(
        tf.keras.layers.Dense(units=512, activation=activation, kernel_regularizer=regularizer)
    )
    model.add(
        tf.keras.layers.Dense(units=256, activation=activation, kernel_regularizer=regularizer)
    )
    model.add(
        tf.keras.layers.Dense(units=128, activation=activation, kernel_regularizer=regularizer)
    )
    model.add(
        tf.keras.layers.Dense(units=64, activation=activation, kernel_regularizer=regularizer)
    )
    model.add(tf.keras.layers.Dense(channels[1]))

    return model


def custom_VGG_V2(patch_size, channels, activation, regularizer):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=(patch_size, patch_size, channels[0])))

    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(2, 2),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(2, 2),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(2, 2),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(
        tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(2, 2),
            padding="valid",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(2, 2),
            padding="valid",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(2, 2),
            padding="valid",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(2, 2),
            padding="valid",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(tf.keras.layers.Flatten())

    model.add(
        tf.keras.layers.Dense(
            units=1024, activation=activation, kernel_regularizer=regularizer
        )
    )
    model.add(
        tf.keras.layers.Dense(units=512, activation=activation, kernel_regularizer=regularizer)
    )
    model.add(
        tf.keras.layers.Dense(units=256, activation=activation, kernel_regularizer=regularizer)
    )
    model.add(
        tf.keras.layers.Dense(units=128, activation=activation, kernel_regularizer=regularizer)
    )

    model.add(tf.keras.layers.Dense(channels[1]))

    return model


def custom_VGG_V3(patch_size, channels, activation, regularizer):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=(patch_size, patch_size, channels[0])))

    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(
        tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(2, 2),
            padding="valid",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(2, 2),
            padding="valid",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(
        tf.keras.layers.Conv2D(
            filters=1024,
            kernel_size=(2, 2),
            padding="valid",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=1024,
            kernel_size=(2, 2),
            padding="valid",
            activation=activation,
            kernel_regularizer=regularizer,
        )
    )

    model.add(tf.keras.layers.Flatten())

    model.add(
        tf.keras.layers.Dense(
            units=1024, activation=activation, kernel_regularizer=regularizer
        )
    )
    model.add(
        tf.keras.layers.Dense(units=512, activation=activation, kernel_regularizer=regularizer)
    )
    model.add(
        tf.keras.layers.Dense(units=512, activation=activation, kernel_regularizer=regularizer)
    )
    model.add(
        tf.keras.layers.Dense(units=256, activation=activation, kernel_regularizer=regularizer)
    )

    model.add(tf.keras.layers.Dense(channels[1]))

    return model


"""
def VGG_V4(patch_size, label_no_cyclone, channels, activation, regularizer):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=(patch_size, patch_size,
    channels[0])))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),
    padding="same", activation=activation, kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),
    padding="same", activation=activation, kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),
    padding="same", activation=activation, kernel_regularizer=regularizer))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),
    padding="same", activation=activation, kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),
    padding="same", activation=activation, kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),
    padding="same", activation=activation, kernel_regularizer=regularizer))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3),
    padding="same", activation=activation, kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3),
    padding="same", activation=activation, kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3),
    padding="same", activation=activation, kernel_regularizer=regularizer))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(2,2),
    padding="same", activation=activation, kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(2,2),
    padding="same", activation=activation, kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(2,2),
    padding="same", activation=activation, kernel_regularizer=regularizer))

    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(2,2),
    padding="valid", activation=activation, kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(2,2),
    padding="valid", activation=activation, kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(2,2),
    padding="valid", activation=activation, kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(2,2),
    padding="valid", activation=activation, kernel_regularizer=regularizer))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(units=1024, activation=activation,
    kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Dense(units=512, activation=activation,
    kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Dense(units=256, activation=activation,
    kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Dense(units=128, activation=activation,
    kernel_regularizer=regularizer))

    model.add(tf.keras.layers.Dense(channels[1]))
    model.add(PositionDiscretizationLayer(label_no_cyclone=label_no_cyclone,
    patch_size=patch_size))

    return model
"""


def ModelV5(patch_size, channels, last_activation, kernel_size=3):
    # kernel initializer
    initializer = tf.random_normal_initializer(0.0, 0.02)

    # input layer
    inputs = tf.keras.layers.Input(shape=(patch_size, patch_size, channels[0]))

    conv_blocks = [
        ConvBlock(
            filters=32,
            initializer=initializer,
            kernel_size=kernel_size,
            strides=2,
            apply_batchnorm=True,
            apply_dropout=False,
            apply_gaussian_noise=True,
        ),
        ConvBlock(
            filters=64,
            initializer=initializer,
            kernel_size=kernel_size,
            strides=2,
            apply_batchnorm=False,
            apply_dropout=False,
            apply_gaussian_noise=False,
        ),
        ConvBlock(
            filters=128,
            initializer=initializer,
            kernel_size=3,
            strides=2,
            apply_batchnorm=False,
            apply_dropout=True,
            apply_gaussian_noise=False,
        ),
        ConvBlock(
            filters=256,
            initializer=initializer,
            kernel_size=3,
            strides=2,
            apply_batchnorm=False,
            apply_dropout=False,
            apply_gaussian_noise=True,
        ),
        ConvBlock(
            filters=512,
            initializer=initializer,
            kernel_size=3,
            strides=2,
            apply_batchnorm=False,
            apply_dropout=False,
            apply_gaussian_noise=False,
        ),
        ConvBlock(
            filters=1024,
            initializer=initializer,
            kernel_size=3,
            strides=2,
            apply_batchnorm=True,
            apply_dropout=True,
            apply_gaussian_noise=False,
        ),
    ]
    x = inputs
    for block in conv_blocks:
        x = block(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1024, activation="relu", kernel_initializer=initializer)(x)
    x = tf.keras.layers.Dense(units=512, activation="relu", kernel_initializer=initializer)(x)
    x = tf.keras.layers.Dense(units=256, activation="relu", kernel_initializer=initializer)(x)
    x = tf.keras.layers.Dense(units=128, activation="relu", kernel_initializer=initializer)(x)

    outputs = tf.keras.layers.Dense(
        channels[1], activation=last_activation, kernel_initializer=initializer
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="model_V5")


"""
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
def custom_VGG_V1(patch_size, channels, activation, regularizer):
    model = Sequential()
    #model.add(Conv2D(input_shape=(40,40,len(variables_list)), filters=64,
    kernel_size=(3,3), padding="same", activation=activation,
    kernel_regularizer=regularizer))
    model.add(Conv2D(input_shape=(patch_size,patch_size,channels[0]),
    filters=64, kernel_size=(3,3), padding="same", activation=activation,
    kernel_regularizer=regularizer))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same",
    activation=activation, kernel_regularizer=regularizer))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same",
    activation=activation, kernel_regularizer=regularizer))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same",
    activation=activation, kernel_regularizer=regularizer))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same",
    activation=activation, kernel_regularizer=regularizer))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(2,2), padding="same",
    activation=activation, kernel_regularizer=regularizer))
    model.add(Conv2D(filters=256, kernel_size=(2,2), padding="same",
    activation=activation, kernel_regularizer=regularizer))
    #model.add(Conv2D(filters=256, kernel_size=(2,2), padding="same",
    activation=activation, kernel_regularizer=regularizer))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(2,2), padding="valid",
    activation=activation, kernel_regularizer=regularizer))
    model.add(Conv2D(filters=512, kernel_size=(2,2), padding="valid",
    activation=activation, kernel_regularizer=regularizer))
    model.add(Conv2D(filters=512, kernel_size=(2,2), padding="valid",
    activation=activation, kernel_regularizer=regularizer))
    model.add(MaxPooling2D(pool_size=(2,2))) #,strides=(2,2)))
    # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same",
    activation="relu"))
    # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same",
    activation="relu"))
    # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same",
    activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=512, activation=activation,
    kernel_regularizer=regularizer))
    model.add(Dense(units=256, activation=activation,
    kernel_regularizer=regularizer))
    model.add(Dense(units=128, activation=activation,
    kernel_regularizer=regularizer))
    model.add(Dense(units=64, activation=activation,
    kernel_regularizer=regularizer))
    model.add(Dense(channels[1]))

    return model
"""
