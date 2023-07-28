import logging
import tensorflow as tf

from ..components import Trainer

class TensorflowTrainer(Trainer):
    def __init__(self, loss, epochs, batch_size, callbacks, optimizer, model_func, metrics_func, strategy=tf.distribute.MirroredStrategy()):
        self.strategy = strategy
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.callbacks = callbacks
        self.optimizer = optimizer

        # Create distributed TF vars
        with self.strategy.scope():
            self.model = model_func()
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=metrics_func())
        print(f"Strategy is working with: {strategy.num_replicas_in_sync} devices")

    def train(self, data):
        (train, n_train), (test, n_test) = data
        #train = self.strategy.experimental_distribute_dataset(train)
        #test = self.strategy.experimental_distribute_dataset(test)

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