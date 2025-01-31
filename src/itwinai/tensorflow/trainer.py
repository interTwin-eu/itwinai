# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Base TensorFlow trainer module."""

from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import tensorflow as tf
from jsonargparse import ArgumentParser
from keras.callbacks import Callback
from tensorflow.data import Dataset

from itwinai.tensorflow.distributed import get_strategy

from ..components import Trainer, monitor_exec


def _import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def _instance_from_dict(obj_dict: Dict, fail_untyped: bool = True) -> Any:
    if isinstance(obj_dict, dict) and obj_dict.get("class_path") is not None:
        # obj_dict is a dictionary with a structure compliant with
        # jsonargparse
        obj_class = _import_class(obj_dict["class_path"])
        parser = ArgumentParser()
        parser.add_subclass_arguments(obj_class, "object", fail_untyped=fail_untyped)
        obj_dict = {"object": obj_dict}
        return parser.instantiate_classes(obj_dict).object

    raise ValueError(
        "Unable to instantiate object with this "
        f"dict configuration: {obj_dict}.\nIt should have "
        "valid 'class_path' and 'init_args' fields"
    )


class TensorflowTrainer(Trainer):
    """Trains a Keras model.

    Args:
        epochs (int): number of training epochs.
        micro_batch_size (int): per-worker batch size. Equals macro batch
            size when not running distributed.
        shuffle_buffer (Optional[int], optional): if given, shuffles
            dataset using a buffer of given size. See
            ``tf.data.Dataset.shuffle``. Defaults to None.
        callbacks (Optional[List], optional): list fo Keras callbacks.
            Can be a list of dictionary configurations. Defaults to None.
        model_config (Optional[Dict], optional): model configuration. If
            given, a model is instantiated from this configuration.
            Defaults to None.
        model_compile_config (Optional[Dict], optional): configuration for
            ``keras.Model.compile``. Defaults to None.
        rnd_seed (Optional[int], optional): random seed. Defaults to None.
        verbose (Union[str, int], optional): verbosity level for
            ``keras.Model.fit``. Defaults to 'auto'.
    """

    #: TensorFlow distributed strategy.
    strategy: tf.distribute.Strategy
    #: Total number of workers in distributed strategy.
    num_workers: int
    #: List of Keras callbacks. Defaults to None.
    callbacks: Optional[List] = None
    #: Total number of training epochs.
    epochs: int
    #: Buffer used to shuffle dataset. Defaults to None.
    shuffle_buffer: Optional[int] = None
    #: Per-worker batch size (when distributed).
    micro_batch_size: int
    #: Total batch size. When distributed, it is the sum of
    #: ``micro_batch_size`` across all workers.
    macro_batch_size: int
    #: Random seed for reproducibility. Defaults to None.
    rnd_seed: Optional[int] = None

    def __init__(
        self,
        epochs: int,
        micro_batch_size: int,
        shuffle_buffer: Optional[int] = None,
        callbacks: Optional[List[Union[Dict, Callback]]] = None,
        model_config: Optional[Dict] = None,
        model_compile_config: Optional[Dict] = None,
        rnd_seed: Optional[int] = None,
        verbose: Union[str, int] = "auto",
    ):
        super().__init__()
        self.save_parameters(**self.locals2params(locals()))
        self.epochs = epochs
        self.micro_batch_size = micro_batch_size
        self.shuffle_buffer = shuffle_buffer
        self.rnd_seed = rnd_seed
        self.verbose = verbose
        if callbacks is not None:
            self.callbacks = self.instantiate_callbacks(callbacks)
        else:
            self.callbacks = []

        # Distributed strategy
        self.strategy, self.num_workers = get_strategy()
        print(f"Distributed strategy is working with: {self.num_workers} devices")
        self.macro_batch_size = self.micro_batch_size * self.num_workers

        # Compile model from configuration, if given
        if model_config is not None and model_compile_config is not None:
            with self.strategy.scope():
                self.model: tf.keras.Model = _instance_from_dict(model_config)
                model_compile_config = self.instantiate_compile_conf(model_compile_config)
                self.model.compile(**model_compile_config)
        else:
            print(
                "Either model_config or model_compile_config were not given. "
                "Skipping automatic model compilation."
            )

    @staticmethod
    def instantiate_compile_conf(model_compile_config: Dict) -> Dict[str, Any]:
        """Instantiate fields of Keras ``model.compile()`` from
        their dictionary serialization.

        Args:
            model_compile_config (Dict): fields of Keras ``model.compile()``
                serialized as dictionary.

        Returns:
            Dict[str, Any]: dictionary mapping compile argument names to
            the instantiated objects.
        """
        final_conf = {}
        for item_name, item in model_compile_config.items():
            if isinstance(item, dict):
                item = _instance_from_dict(item)
            final_conf[item_name] = item
        return final_conf

    @staticmethod
    def instantiate_callbacks(callbacks: List[Union[Dict, Callback]]) -> List[Callback]:
        """Instantiate Keras callbacks from dictionaries.

        Args:
            callbacks (List[Union[Dict, Callback]]): list of Keras callbacks
            in serialized as dictionary.

        Returns:
            List[Callback]: list of instantiated callbacks.
        """
        final_callbacks = []
        for item in callbacks:
            if isinstance(item, dict):
                # Not all constructor args in keras callbacks
                # are typed!
                item = _instance_from_dict(item, fail_untyped=False)
            final_callbacks.append(item)
        return final_callbacks

    @monitor_exec
    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
    ) -> Tuple[Dataset, Dataset, Dataset, keras.Model]:
        """Run training. Users should override this method.

        Args:
            train_dataset (Dataset): train dataset of type
                ``tensorflow.data.Dataset``.
            validation_dataset (Dataset): validation dataset of type
                ``tensorflow.data.Dataset``.
            test_dataset (Optional[Dataset], optional): test dataset
                of type ``tensorflow.data.Dataset``. Defaults to None.

        Returns:
            Tuple[Dataset, Dataset, Dataset, keras.Model]: tuple of
            train_dataset, validation_dataset, test_dataset, and trained
            Keras model.
        """

        print(f"len(train_dataset): {len(train_dataset)}")
        print(f"len(validation_dataset): {len(validation_dataset)}")
        print("micro_batch_size: ", self.micro_batch_size, flush=True)
        print("macro_batch_size: ", self.macro_batch_size, flush=True)

        # Shuffle dataset
        if self.shuffle_buffer:
            train_ds = train_dataset.shuffle(self.shuffle_buffer, seed=self.rnd_seed)
            valid_ds = validation_dataset.shuffle(self.shuffle_buffer, seed=self.rnd_seed)
        else:
            train_ds = train_dataset
            valid_ds = validation_dataset

        # Set batch size to the dataset and repeat
        train_ds = train_ds.batch(
            self.macro_batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
        ).repeat(self.epochs)
        valid_ds = valid_ds.batch(
            self.macro_batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
        ).repeat(self.epochs)

        print(f"len(train_ds): {len(train_ds)}")
        print(f"len(valid_ds): {len(valid_ds)}")

        # Distribute datasets among strategy's replica
        dist_train_dataset = self.strategy.experimental_distribute_dataset(train_ds)
        dist_valid_dataset = self.strategy.experimental_distribute_dataset(valid_ds)

        print(f"len(dist_train_dataset): {len(train_ds)}")
        print(f"len(dist_train_dataset): {len(valid_ds)}")

        # Compute the steps per epoch for train and valid
        steps_per_epoch = len(train_dataset) // self.macro_batch_size
        validation_steps = len(validation_dataset) // self.macro_batch_size

        print(f"steps_per_epoch: {steps_per_epoch}")
        print(f"validation_steps: {validation_steps}")

        #####################################################################
        # Instantiate here model, optimizer, loss under the strategy scope, #
        # if not done previously through `model_compile_config` and         #
        # `model_config` !                                                  #
        # Always remember that they should be instantiated under the        #
        # distributed strategy scope: ``with self.strategy.scope():``       #
        #                                                                   #
        # Example:                                                          #
        # with self.strategy.scope():                                       #
        #   model = tf.keras.Sequential(...)                                #
        #   optimizer = rf.keras.optimizers.Adam(...)                       #
        #   loss = tf.keras.losses.BinaryCrossentropy(...)                  #
        #####################################################################

        # Train the model
        self.model.fit(
            dist_train_dataset,
            validation_data=dist_valid_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=self.verbose,
        )
        print("Training completed")
        return train_dataset, validation_dataset, test_dataset, self.model
