# This file contains TFX functions that will be run internally
import tensorflow as tf
import tensorflow_transform as tft

from typing import List
from model import MNISTModel

from tfx import v1 as tfx
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options

# MNIST Dataset keys
IMAGE_KEY = 'image'
LABEL_KEY = 'label'

# Keys transforms
def transformed_name(key):
  return key + '_xf'

# Serve examples for tensorflow
def serve_tf_examples(model, tf_transform_output):
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    transformed_features = model.tft_layer(parsed_features)
    return model(transformed_features)

  return serve_tf_examples_fn

# Internally run function by TFX for training procedure
def run_fn(fn_args: tfx.components.FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, fn_args.data_accessor, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, fn_args.data_accessor, tf_transform_output)

    model = MNISTModel(transformed_name(IMAGE_KEY))
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps
    )

    signatures = {
        'serving_default':
            serve_tf_examples(
                model, tf_transform_output).get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
            )
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

# Internally run function by TFX for preprocessing procedure
def preprocessing_fn(inputs):
    img = tf.map_fn(
        fn=lambda x: tf.io.parse_tensor(x[0], tf.uint8, name=None),
        elems=inputs[IMAGE_KEY],
        fn_output_signature=tf.TensorSpec((28, 28), dtype=tf.uint8, name=None),
        infer_shape=True
    )
    img = tf.cast(img, tf.int64)
    outputs = {transformed_name(IMAGE_KEY): img, transformed_name(LABEL_KEY): inputs[LABEL_KEY]}
    return outputs

# Internally run function by TFX for input transform procedure
def input_fn(
        file_pattern: List[str],
        data_accessor: DataAccessor,
        tf_transform_output: tft.TFTransformOutput,
        batch_size: int = 20
) -> tf.data.Dataset:
    return data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size,
          label_key=transformed_name(LABEL_KEY)
      ),
      tf_transform_output.transformed_metadata.schema
    ).repeat()

