import tensorflow as tf

from ..scaling import minmax_transform


def get_tensor_decoding_fn(
    shape, drv_vars=[], coo_vars=None, msk_var=None,
    dtype=tf.float32
):

    def tensor_decoding_fn(serialized_data):
        """ Decoding function for a dataset written to disk as
        tensor_encoding_fn().
        """
        # define features dictionary
        features = {}

        # define all variables list
        vars = drv_vars.copy()
        if coo_vars:
            vars += coo_vars.copy()
        if msk_var:
            vars += [msk_var]

        # add driver + coordinate + mask vars to features
        for var in vars:
            features.update({var: tf.io.FixedLenFeature([], tf.string)})

        # parse the serialized data so we get a dict with our data.
        parsed_data = tf.io.parse_single_example(
            serialized_data, features=features)

        # accumulator for data elements
        data = []

        # get x raw data
        Xdrv = tf.stack([tf.ensure_shape(tf.io.parse_tensor(
            serialized=parsed_data[var], out_type=dtype),
            shape=shape)for var in drv_vars], axis=-1)
        data.append(Xdrv)

        # if coordinate vars are provided
        if coo_vars:
            if len(coo_vars) == 1:
                Ycoo = tf.ensure_shape(tf.io.parse_tensor(
                    serialized=parsed_data[coo_vars[0]], out_type=dtype),
                    shape=(2,))
            else:
                Ycoo = tf.stack([tf.ensure_shape(tf.io.parse_tensor(
                    serialized=parsed_data[var], out_type=dtype),
                    shape=(2)) for var in coo_vars], axis=-1)
            data.append(Ycoo)

        # if mask var is provided
        if msk_var:
            Ymsk = tf.expand_dims(tf.ensure_shape(tf.io.parse_tensor(
                serialized=parsed_data[msk_var], out_type=dtype),
                shape=shape), axis=-1)
            data.append(Ymsk)

        return tuple(data)

    return tensor_decoding_fn


def get_resize_fn(shape):

    def resize_fn(data):
        """Resize function that resizes the input data to the target shape."""
        resized_data = []
        for x in data:
            resized_data.append(tf.image.resize(
                x, shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        return tuple(resized_data)

    return resize_fn


def get_scaling_fn(scalers):

    def scaling_fn(data):
        """Function to scale the data according to the provided scalers."""
        data_scaled = []
        for x, scaler in zip(data, scalers):
            data_scaled.append(minmax_transform(x, scaler))
        return tuple(data_scaled)

    return scaling_fn


def get_scale_target_fn(label_no_cyclone, patch_size):

    def scale_target_fn(data):
        """Function to scale the target according to the provided
        label_no_cyclone"""
        x, y = data
        # scale y
        y_scaled = tf.math.divide(
            tf.subtract(tf.cast(y, dtype=tf.float32), label_no_cyclone),
            tf.subtract(tf.cast(patch_size-1, dtype=tf.float32),
                        label_no_cyclone)
        )
        return (x, y_scaled)

    return scale_target_fn


def get_masking_fn(mask):

    def masking_fn(data):
        # TODO : parametrize this function as the others
        X, y = data
        y_masked = tf.where(y < 0, mask, y)
        return (X, y_masked)

    return masking_fn


def read_tfrecord_as_tensor(filenames, shape, drv_vars, coo_vars, msk_var):
    # set autotune parameter to automatically manage resourches
    AUTOTUNE = tf.data.AUTOTUNE

    # get lambda functions to be applied to this dataset
    tensor_decoding_fn = get_tensor_decoding_fn(
        shape, drv_vars=drv_vars, coo_vars=coo_vars, msk_var=msk_var)

    # compute the number of samples into the dataset
    n_elems = sum(1 for _ in tf.data.TFRecordDataset(filenames))

    # Create standard dataset
    dataset = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=AUTOTUNE).map(
        tensor_decoding_fn, num_parallel_calls=AUTOTUNE)

    # read data as numpy
    Xdata, ydata = dataset.batch(batch_size=n_elems).as_numpy_iterator().next()
    Xt = tf.convert_to_tensor(Xdata)
    yt = tf.convert_to_tensor(ydata)

    return Xt, yt
