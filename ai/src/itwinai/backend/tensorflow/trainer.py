import logging
import tensorflow as tf

from ..components import Trainer


class TensorflowTrainer(Trainer):
    def __init__(
            self,
            epochs,
            batch_size,
            callbacks,
            model_func,
            compile_conf,
            strategy
    ):
        self.strategy = strategy
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks

        # Create distributed TF vars
        if self.strategy:
            with self.strategy.scope():
                self.model = model_func()
                self.model.compile(compile_conf)
        else:
            self.model = model_func()
            self.model.compile(compile_conf)

        num_devices = self.strategy.num_replicas_in_sync if self.strategy else 1
        print(f"Strategy is working with: {num_devices} devices")

    def train(self, data):
        train, test = data

        # Set batch size to the dataset
        train = train.batch(self.batch_size)
        test = test.batch(self.batch_size)

        # Extract number of samples for each dataset
        n_train = train.cardinality().numpy()
        n_test = test.cardinality().numpy()

        # Distribute dataset
        if self.strategy:
            train = self.strategy.experimental_distribute_dataset(train)
            test = self.strategy.experimental_distribute_dataset(test)

        # compute the steps per epoch for train and valid
        train_steps = n_train // self.batch_size
        test_steps = n_test // self.batch_size

        # train the model
        self.model.fit(
            train,
            validation_data=test,
            steps_per_epoch=train_steps,
            validation_steps=test_steps,
            epochs=self.epochs,
            callbacks=self.callbacks,
        )

        logging.debug(f"Model trained")

    # Executable
    def execute(self, args):
        raise "Not implemented!"

    def setup(self, args):
        raise "Not implemented!"