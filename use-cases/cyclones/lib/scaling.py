from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import joblib


def fit_transform(
    volume, shape, channel, feature_range=(0, 1),
        type='minmax', save=False, filename=None
):
    """
    Creates the scaler on the input volume and scales the data

    Parameters
    ----------
    volume : np.array
        Input 4-dimensional data volume.
    shape : (int, int)
        Height and width of the input volume.
    channel : int
        Number of channels of the input volume.
    feature_range : (int, int) | (0,1)
        Desired range of the scaled features. Default is (0,1).
    type : {'minmax', 'std', ...}
        Type of scaling. Default MinMax scaler
    save : bool | False
        Whether or not to save the computed scaler. Default to false
    filename : pathlike
        The filename to which the scaler must be saved to disk. Checked only
        if 'save' is set to True.

    Returns
    -------
    volume : np.array
        Scaled 4-dimensional input volume.
    scaler : scikit-learn scaler
        Computed scaler.
    """

    can_save = False
    if save:
        if filename is None:
            raise ValueError(
                'Must specify the filename when saving the scaler')
        else:
            can_save = True

    if type == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)

    volume = scaler.fit_transform(
        volume.reshape(-1, channel)).reshape(-1, *shape, channel)

    if can_save:
        joblib.dump(scaler, filename)

    return volume, scaler


def inv_transform(scaled_image, scaler, shape, channel):
    """
    Scale the data from scaler's feature_range to data range

    Parameters
    ----------
    scaled_image : np.array
        4-dimensional input data volume.
    scaler : scikit-learn scaler
        Desired data scaler.
    shape : tuple
        (int, int) height and width of the input volume.
    channel : int
        Number of channels of the input volume.

    Returns
    -------
    image : np.array
        Scaled input volume.
    """
    return scaler.inverse_transform(
        scaled_image.reshape(-1, channel)
    ).reshape(*shape)


def transform(image, scaler, shape, channel):
    """
    Scale the data from data range to scaler's feature range

    Parameters
    ----------
    image : np.array
        4-dimensional input data volume.
    scaler : scikit-learn scaler
        Desired data scaler.
    shape : tuple
        (int, int) height and width of the input volume.
    channel : int
        Number of channels of the input volume.

    Returns
    -------
    scaled_image : np.array
        Scaled input volume.
    """
    return scaler.transform(
        image.reshape(-1, channel)
    ).reshape(-1, *shape, channel)


def get_scalers(scaler_X_file=None, scaler_y_file=None):
    """
    Reads the X and y scalers from file
    Input:
        @ (string) scaler_X_file : X data scaler of joblib.
            None if data must not be scaled
        @ (string) scaler_y_file : y data scaler of joblib.
            None if data must not be scaled
    Output:
        @ ([scaler,scaler]) scalers : loaded data scalers
    """
    X_scaler = None
    y_scaler = None
    if scaler_X_file:
        X_scaler = joblib.load(scaler_X_file)
    if scaler_y_file:
        y_scaler = joblib.load(scaler_y_file)
    return [X_scaler, y_scaler]


def save_tf_minmax(Xt, outfile):
    """
    Saves a MinMax Scaler as a Tensorflow Record.

    """
    def tensor_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()]
        )
        )

    def scaler_encoding_fn(min, max):
        """Builds a serialized version of the dataset. X and y
        must be np.array.
        """
        features = tf.train.Features(feature={
            "min": tensor_feature(min),
            "max": tensor_feature(max),
        })
        return tf.train.Example(features=features).SerializeToString()

    def write_record_to_file(min, max, record_file):
        with tf.io.TFRecordWriter(record_file) as writer:
            record = scaler_encoding_fn(min=min, max=max)
            writer.write(record)
        return

    # passed Xt is a numpy array of shape N x H x W x C
    batch_size = 8192
    n_batches = Xt.shape[0] // batch_size
    if Xt.shape[0] % batch_size:
        n_batches += 1

    # compute min
    cur_min = tf.math.reduce_min(input_tensor=Xt[0, ], axis=(0, 1)).numpy()
    for i in range(n_batches):
        X_batch = Xt[(i * batch_size):((i+1) * batch_size)]
        i_min = tf.math.reduce_min(
            input_tensor=X_batch, axis=(0, 1, 2)).numpy()
        for c in range(i_min.shape[-1]):
            if i_min[c] <= cur_min[c]:
                cur_min[c] = i_min[c]
    X_min = cur_min

    # compute max
    cur_max = tf.math.reduce_max(input_tensor=Xt[0, ], axis=(0, 1)).numpy()
    for i in range(n_batches):
        X_batch = Xt[(i * batch_size):((i+1) * batch_size)]
        i_max = tf.math.reduce_max(
            input_tensor=X_batch, axis=(0, 1, 2)).numpy()
        for c in range(i_max.shape[-1]):
            if i_max[c] >= cur_max[c]:
                cur_max[c] = i_max[c]
    X_max = cur_max

    # write min and max to file
    if outfile:
        write_record_to_file(min=X_min, max=X_max, record_file=outfile)

    # return scaler dictionary
    return {'min': tf.convert_to_tensor(X_min),
            'max': tf.convert_to_tensor(X_max)}


def save_tf_minmax_by_min_and_max(min, max, outfile):
    """
    Saves a MinMax Scaler as a Tensorflow Record.
    """
    def tensor_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()]))

    def scaler_encoding_fn(min, max):
        """Builds a serialized version of the dataset. X and y must be
        np.array.
        """
        features = tf.train.Features(feature={
            "min": tensor_feature(min),
            "max": tensor_feature(max),
        })
        return tf.train.Example(features=features).SerializeToString()

    def write_record_to_file(min, max, record_file):
        with tf.io.TFRecordWriter(record_file) as writer:
            record = scaler_encoding_fn(min=min, max=max)
            writer.write(record)
        return

    # write min and max to file
    if outfile:
        write_record_to_file(min=min, max=max, record_file=outfile)

    # return scaler dictionary
    return {'min': tf.convert_to_tensor(min), 'max': tf.convert_to_tensor(max)}


def load_tf_minmax(scalerfile, vars):
    """
    Loads a MinMax Scaler from disk as a Tensorflow Record.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    def scaler_decoding_fn(serialized_data):
        """Decoding function for a dataset written to disk as
        tensor_encoding_fn().
        """
        features = {
            'min': tf.io.FixedLenFeature([], tf.string),
            'max': tf.io.FixedLenFeature([], tf.string)
        }
        # Parse the serialized data so we get a dict with our data.
        parsed_data = tf.io.parse_single_example(
            serialized_data, features=features)
        # Get X and y raw data
        raw_min = parsed_data['min']
        raw_max = parsed_data['max']
        # Decode the raw bytes so it becomes a tensor with type.
        min = tf.ensure_shape(tf.io.parse_tensor(
            raw_min, tf.float32), (len(vars)))
        max = tf.ensure_shape(tf.io.parse_tensor(
            raw_max, tf.float32), (len(vars)))
        return min, max

    # load scaler set
    scaler_set = (
        tf.data.TFRecordDataset(scalerfile, num_parallel_reads=AUTOTUNE)
        .map(scaler_decoding_fn, num_parallel_calls=AUTOTUNE)
    )
    # get min and max from the dataset
    for data in scaler_set:
        min, max = data

    # return scaler dictionary
    return {'min': min, 'max': max}


def minmax_transform(data, scaler):
    """
    Applies the transform of TFMinMaxScaler to the provided dataset.
    """
    if scaler:
        num = tf.subtract(data, scaler['min'])
        den = tf.subtract(scaler['max'], scaler['min'])
        res = tf.math.divide(num, den)
    else:
        res = data
    return res


def minmax_inverse_transform(scaled_data, scaler):
    """
    Applies the inverse transform of TFMinMaxScaler to the provided
    scaled dataset.
    """
    sub = tf.subtract(scaler['max'], scaler['min'])
    mul = tf.multiply(scaled_data, sub)
    return mul + scaler['min']


def minmax_inverse_target_transform(y_scaled, label_no_cyclone, patch_size):
    """
    Applies the inverse transform on y data when scaled in (0,1)
    """
    sub = tf.subtract(
        tf.cast(patch_size-1, dtype=tf.float32), label_no_cyclone)
    mul = tf.multiply(y_scaled, sub)
    return mul + label_no_cyclone
