from typing import Optional, Dict, Tuple
import tensorflow.keras as keras
import tensorflow as tf

from itwinai.backend.components import DataGetter, DataPreproc


class MNISTDataGetter(DataGetter):
    def __init__(self):
        super().__init__()

    def load(self):
        return keras.datasets.mnist.load_data()

    def execute(
        self,
        config: Optional[Dict] = None
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        train, test = self.load()
        return ([train, test],), config


class MNISTDataPreproc(DataPreproc):
    def __init__(self, classes: int):
        super().__init__()
        self.classes = classes

    def preproc(self, datasets) -> Tuple:
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

    def execute(
        self,
        datasets,
        config: Optional[Dict] = None
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        return self.preproc(datasets), config
