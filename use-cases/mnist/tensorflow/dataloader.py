from typing import Tuple
import tensorflow.keras as keras
import tensorflow as tf

from itwinai.components import DataGetter, DataPreproc, monitor_exec


class MNISTDataGetter(DataGetter):
    def __init__(self):
        super().__init__()
        self.save_parameters(**self.locals2params(locals()))

    @monitor_exec
    def execute(self) -> Tuple:
        train, test = keras.datasets.mnist.load_data()
        return train, test


class MNISTDataPreproc(DataPreproc):
    def __init__(self, classes: int):
        super().__init__()
        self.save_parameters(**self.locals2params(locals()))
        self.classes = classes

    @monitor_exec
    def execute(
        self,
        *datasets,
    ) -> Tuple:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.FILE)
        preprocessed = []
        for dataset in datasets:
            x, y = dataset
            y = keras.utils.to_categorical(y, self.classes)
            sliced = tf.data.Dataset.from_tensor_slices((x, y))
            sliced = sliced.with_options(options)
            preprocessed.append(sliced)
        return tuple(preprocessed)
