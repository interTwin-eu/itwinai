import tensorflow.keras as keras

from itwinai.backend.components import DataGetter, DataPreproc


class TensorflowDataGetter(DataGetter):
    def __init__(self):
        pass

    def load(self):
        return keras.datasets.mnist.load_data()

    def execute(self, args):
        train, test = self.load()
        return [train, test]

    def setup(self, args):
        pass


class TensorflowDataPreproc(DataPreproc):
    def __init__(self, classes: int):
        self.classes = classes

    def preproc(self, datasets):
        preprocessed = []
        for dataset in datasets:
            x, y = dataset
            y = keras.utils.to_categorical(y, self.classes)
            preprocessed.append((x, y))
        return preprocessed

    def execute(self, datasets):
        return self.preproc(datasets)

    def setup(self, args):
        pass
