import tensorflow.keras as keras

# TODO: Solve relative import
import sys
sys.path.append("..")
from components import DataGetter, DataPreproc

from marshmallow_dataclass import dataclass

@dataclass
class DataGetterConf:
    pass

class TensorflowDataGetter(DataGetter):
    def __init__(self):
        pass

    def load(self):
        return keras.datasets.mnist.load_data()

    def execute(self, args):
        train, test = self.load()
        return [train, test]

    def config(self, config):
        pass

@dataclass
class DataPreprocConf:
    classes: int

class TensorflowDataPreproc(DataPreproc):
    def __init__(self):
        self.classes = None

    def preproc(self, datasets):
        preprocessed = []
        for dataset in datasets:
            x, y = dataset
            y = keras.utils.to_categorical(y, self.classes)
            preprocessed.append((x, y))
        return preprocessed

    def execute(self, datasets):
        return self.preproc(datasets)

    def config(self, config):
        config = DataPreprocConf.Schema().load(config['DataPreprocConf'])
        self.classes = config.classes