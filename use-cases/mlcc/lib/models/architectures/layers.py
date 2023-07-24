import tensorflow as tf



def ConvBlock(filters, initializer, kernel_size=3, strides=1, apply_batchnorm=False, apply_dropout=False, apply_gaussian_noise=False):
    layer = tf.keras.Sequential()
    layer.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer=initializer, use_bias=False))
    if apply_gaussian_noise:
        layer.add(tf.keras.layers.GaussianNoise(stddev=1.0))
    if apply_batchnorm:
        layer.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        layer.add(tf.keras.layers.Dropout(0.5))
    layer.add(tf.keras.layers.LeakyReLU())
    return layer



def EncoderBlock(filters, kernel_size, strides=2, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    layer = tf.keras.Sequential()
    layer.add(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        layer.add(tf.keras.layers.BatchNormalization())
    layer.add(tf.keras.layers.LeakyReLU())
    return layer



def DecoderBlock(filters, kernel_size, strides=2, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    layer = tf.keras.Sequential()
    layer.add(tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', kernel_initializer=initializer, use_bias=False))
    layer.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        layer.add(tf.keras.layers.Dropout(0.5))
    layer.add(tf.keras.layers.ReLU())
    return layer


