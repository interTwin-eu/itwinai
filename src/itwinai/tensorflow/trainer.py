"""Base TensorFlow trainer module."""
from typing import Dict, Any

from jsonargparse import ArgumentParser
import tensorflow as tf

from ..components import Trainer, monitor_exec
from itwinai.tensorflow.distributed import get_strategy


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def instance_from_dict(obj_dict: Any) -> Any:
    if isinstance(obj_dict, dict) and obj_dict.get('class_path') is not None:
        # obj_dict is a dictionary with a structure compliant with
        # jsonargparse
        obj_class = import_class(obj_dict["class_path"])
        parser = ArgumentParser()
        parser.add_subclass_arguments(obj_class, "object")
        obj_dict = {"object": obj_dict}
        return parser.instantiate_classes(obj_dict).object
    return obj_dict


class TensorflowTrainer(Trainer):
    def __init__(
            self,
            epochs,
            train_dataset,
            validation_dataset,
            batch_size,
            callbacks,
            model_dict: Dict,
            compile_conf,
            strategy
    ):
        super().__init__()
        self.save_parameters(**self.locals2params(locals()))
        self.strategy = strategy
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks

        # # Handle the parsing
        # model_class = import_class(model_dict["class_path"])
        # parser = ArgumentParser()
        # parser.add_subclass_arguments(model_class, "model")
        # model_dict = {"model": model_dict}

        # from itwinai.models.tensorflow.mnist import MNIST_Model

        # Create distributed TF vars
        if self.strategy:
            tf_dist_strategy, n_devices = get_strategy()
            # get total number of workers
            print("Number of devices: {}".format(n_devices))
            # distribute datasets among MirroredStrategy's replicas
            dist_train_dataset = (
                tf_dist_strategy.experimental_distribute_dataset(
                    train_dataset
                ))
            dist_validation_dataset = (
                tf_dist_strategy.experimental_distribute_dataset(
                    validation_dataset
                ))
            with self.strategy.scope():
                # TODO: move loss, optimizer and metrics instantiation under
                # here
                # Ref:
                # https://www.tensorflow.org/guide/distributed_training#use_tfdistributestrategy_with_keras_modelfit
                # self.model: tf.keras.Model = parser.instantiate_classes(
                #     model_dict).model
                # TODO: add dataloaders and model instances
                self.model: tf.keras.Model = instance_from_dict(model_dict)
                compile_conf = self.instantiate_compile_conf(compile_conf)
                self.model.compile(**compile_conf)
                # print(self.model)
                # self.model.compile(**compile_conf)

                # self.model = tf.keras.Sequential([
                #     tf.keras.layers.Conv2D(
                #         32, 3, activation='relu', input_shape=(28, 28, 1)),
                #     tf.keras.layers.MaxPooling2D(),
                #     tf.keras.layers.Flatten(),
                #     tf.keras.layers.Dense(64, activation='relu'),
                #     tf.keras.layers.Dense(10)
                # ])
                # self.model = MNIST_Model()
                # self.model.compile(loss='mse', optimizer='sgd')
                # self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                #                    optimizer=tf.keras.optimizers.Adam(
                #     learning_rate=0.001),
                #     metrics=['accuracy'])

        else:
            self.model: tf.keras.Model = instance_from_dict(model_dict)
            compile_conf = self.instantiate_compile_conf(compile_conf)
            self.model.compile(**compile_conf)

        self.num_devices = (
            self.strategy.num_replicas_in_sync if self.strategy else 1)
        print(f"Strategy is working with: {self.num_devices} devices")

    @staticmethod
    def instantiate_compile_conf(conf: Dict) -> Dict:
        for item_name, item in conf.items():
            conf[item_name] = instance_from_dict(item)
        return conf

    @monitor_exec
    def execute(self, train_dataset, validation_dataset) -> Any:
        # Set batch size to the dataset
        # train = train.batch(self.batch_size, drop_remainder=True)
        # test = test.batch(self.batch_size, drop_remainder=True)

        # Number of samples
        # n_train = train.cardinality().numpy()
        # n_test = test.cardinality().numpy()
        print(
            f"TRAIN CARD: {train_dataset.cardinality().numpy()} - "
            f"LEN: {len(train_dataset)}")
        print(next(iter(train_dataset)))
        print(type(train_dataset))
        # n_train = len(train) // self.batch_size
        # n_test = len(test) // self.batch_size

        # TODO: read
        # https://github.com/tensorflow/tensorflow/issues/56773#issuecomment-1188693881
        # https://www.tensorflow.org/guide/distributed_training#use_tfdistributestrategy_with_keras_modelfit

        # # Distribute dataset
        # if self.strategy:
        #     train = self.strategy.experimental_distribute_dataset(train)
        #     test = self.strategy.experimental_distribute_dataset(test)

        assert isinstance(train_dataset, tf.data.Dataset)

        # train the model
        history = self.model.fit(
            train_dataset.batch(self.batch_size),
            validation_data=validation_dataset.batch(self.batch_size),
            # steps_per_epoch=int(n_train // self.num_devices),
            # validation_steps=int(n_test // self.num_devices),
            epochs=self.epochs,
            callbacks=self.callbacks,
            # batch_size=self.batch_size
        )

        print("Model trained")
        return history


# class TensorflowTrainer2(Trainer):
#     def __init__(
#             self,
#             epochs,
#             batch_size,
#             callbacks,
#             model_dict: Dict,
#             compile_conf,
#             strategy
#     ):
#         super().__init__()
#         self.strategy = strategy
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.callbacks = callbacks

#         # Handle the parsing
#         model_class = import_class(model_dict["class_path"])
#         parser = ArgumentParser()
#         parser.add_subclass_arguments(model_class, "model")
#         model_dict = {"model": model_dict}

#         # Create distributed TF vars
#         if self.strategy:
#             with self.strategy.scope():
#                 self.model = parser.instantiate_classes(model_dict).model
#                 print(self.model)
#                 self.model.compile(**compile_conf)
#                 # TODO: move loss, optimizer and metrics instantiation under
#                 # here
#                 # Ref:
#                 # https://www.tensorflow.org/guide/distributed_training\
#                   #use_tfdistributestrategy_with_keras_modelfit
#         else:
#             self.model = parser.instantiate_classes(model_dict).model
#             self.model.compile(**compile_conf)

#         self.num_devices = (
#             self.strategy.num_replicas_in_sync if self.strategy else 1)
#         print(f"Strategy is working with: {self.num_devices} devices")

#     def train(self, train_dataset, validation_dataset):
#         # TODO: FIX Steps sizes in model.fit
#         train, test = train_dataset, validation_dataset

#         # Set batch size to the dataset
#         train = train.batch(self.batch_size, drop_remainder=True)
#         test = test.batch(self.batch_size, drop_remainder=True)

#         # Number of samples
#         n_train = train.cardinality().numpy()
#         n_test = test.cardinality().numpy()

#         # TODO: read
#         # https://github.com/tensorflow/tensorflow/issues/56773\
#           #issuecomment-1188693881
#         # https://www.tensorflow.org/guide/distributed_training\
#           #use_tfdistributestrategy_with_keras_modelfit

#         # Distribute dataset
#         if self.strategy:
#             train = self.strategy.experimental_distribute_dataset(train)
#             test = self.strategy.experimental_distribute_dataset(test)

#         # train the model
#         history = self.model.fit(
#             train,
#             validation_data=test,
#             steps_per_epoch=int(n_train // self.num_devices),
#             validation_steps=int(n_test // self.num_devices),
#             epochs=self.epochs,
#             callbacks=self.callbacks,
#         )

#         print("Model trained")
#         return history
