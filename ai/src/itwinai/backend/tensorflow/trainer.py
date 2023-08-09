import logging
import importlib
import tensorflow as tf
import argparse

from jsonargparse import ArgumentParser
from ..components import Trainer
from itwinai.models.tensorflow.cyclegan import CycleGAN

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class TensorflowTrainer(Trainer):
    def __init__(
            self,
            epochs,
            batch_size,
            callbacks,
            model_dict,
            compile_conf,
            strategy
    ):
        self.strategy = strategy
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks

        # Handle the parsing
        model_class = import_class(model_dict["class_path"])
        parser = ArgumentParser()
        parser.add_subclass_arguments(model_class, "model")
        model_dict = {"model": model_dict}

        # Create distributed TF vars
        if self.strategy:
            with self.strategy.scope():
                self.model = parser.instantiate_classes(model_dict).model
                self.model.compile(compile_conf)
        else:
            self.model = parser.instantiate_classes(model_dict).model
            self.model.compile(compile_conf)

        self.num_devices = self.strategy.num_replicas_in_sync if self.strategy else 1
        print(f"Strategy is working with: {self.num_devices} devices")

    def train(self, data):
        # TODO: Batch_per_worker? Validation_steps? Train_steps?
        train, test = data

        # Set batch size to the dataset
        train = train.batch(self.batch_size * self.num_devices, drop_remainder=True)
        test = test.batch(self.batch_size * self.num_devices, drop_remainder=True)

        # Distribute dataset
        if self.strategy:
            train = self.strategy.experimental_distribute_dataset(train)
            test = self.strategy.experimental_distribute_dataset(test)

        # train the model
        self.model.fit(
            train,
            validation_data=test,
            steps_per_epoch=int(len(train) // self.batch_size),
            validation_steps=int(len(test) // self.batch_size),
            epochs=self.epochs,
            callbacks=self.callbacks,
        )

        logging.debug(f"Model trained")

    # Executable
    def execute(self, args):
        raise "Not implemented!"

    def setup(self, args):
        raise "Not implemented!"