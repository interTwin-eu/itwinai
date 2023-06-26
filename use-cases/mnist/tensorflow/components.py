# This file contains TFX functions that will be run internally
import tensorflow as tf
import tensorflow_transform as tft

from typing import List
from model import MNISTModel

from tfx import v1 as tfx
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options

"""
File for Components required for TFX
"""

# MNIST Dataset keys
IMAGE_KEY = 'image'
LABEL_KEY = 'label'

# Keys transforms
def transform_name(key):
    """
    Transforms name with the given key
    """
    return key + '_xf'

def serve_example(model, transformed_output):
    """
    Serve the examples for tensorflow signatures
    """
    model.tft_layer = transformed_output.transform_features_layer()

    @tf.function
    def serve_example_fn(serialized_tf_examples):
        """
        Gives output to be used for signatures
        """
        features = transformed_output.raw_feature_spec()
        features.pop(LABEL_KEY)
        features_transformer = model.tft_layer(tf.io.parse_example(serialized_tf_examples, features))
        return model(features_transformer)

    return serve_example_fn

def run_fn(fn_args: tfx.components.FnArgs):
    """
    Main function for the training that is internally run by the TFX modules
    """
    transformed_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, fn_args.data_accessor, transformed_output)
    eval_dataset = input_fn(fn_args.eval_files, fn_args.data_accessor, transformed_output)

    model = MNISTModel(transform_name(IMAGE_KEY))
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps
    )

    signatures = {
        'serving_default':
            serve_example(
                model, transformed_output).get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
            )
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

def preprocessing_fn(inputs):
    """
    Preprocessing function for the MNIST tfrecords that is run internally in the TFX
    """
    img = tf.map_fn(
        fn=lambda x: tf.io.parse_tensor(x[0], tf.uint8, name=None),
        elems=inputs[IMAGE_KEY],
        fn_output_signature=tf.TensorSpec((28, 28), dtype=tf.uint8, name=None),
        infer_shape=True
    )
    img = tf.cast(img, tf.int64)
    outputs = {transform_name(IMAGE_KEY): img, transform_name(LABEL_KEY): inputs[LABEL_KEY]}
    return outputs

def input_fn(
        file_pattern: List[str],
        data_accessor: DataAccessor,
        transformed_output: tft.TFTransformOutput,
        batch_size: int = 20
) -> tf.data.Dataset:
    """
    Internally run function that batches and transforms the data, provides input for the TFX pipeline
    """
    return data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size,
          label_key=transform_name(LABEL_KEY)
      ),
      transformed_output.transformed_metadata.schema
    ).repeat()

