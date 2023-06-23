import os
import tensorflow_model_analysis as tfma
import argparse

from tfx import v1 as tfx


def _create_pipeline(
        pipeline_name: str,
        pipeline_root: str,
        train_data_root: str,
        module_file: str,
        serving_model_dir: str,
        metadata_path: str
) -> tfx.dsl.Pipeline:
  # Brings data into the pipeline.
  train_gen = tfx.components.ImportExampleGen(input_base=train_data_root).with_id('train_get')

  # Computes statistics over data for visualization and example validation.
  statistics_gen = tfx.components.StatisticsGen(examples=train_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = tfx.components.SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

  # Performs transformations and feature engineering in training and serving.
  transform = tfx.components.Transform(
      examples=train_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file
  )

  # Uses user-provided Python function that trains a model.
  trainer = tfx.components.Trainer(
      module_file=module_file,
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      train_args=tfx.proto.TrainArgs(num_steps=10000),
      eval_args=tfx.proto.EvalArgs(num_steps=5)
  )

  # Define config for the evaluator
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key='label')],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='SparseCategoricalAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0.8}
                      )
                  )
              )
          ])
      ])

  # Use evaluator to evaluate model using tfx
  evaluator = tfx.components.Evaluator(
      examples=train_gen.outputs['examples'],
      model=trainer.outputs['model'],
      eval_config=eval_config
  ).with_id('Evaluator.mnist')

  # Pushes the model to a filesystem destination.
  pusher = tfx.components.Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=tfx.proto.PushDestination(
          filesystem=tfx.proto.PushDestination.Filesystem(
              base_directory=serving_model_dir))
  )

  # Following three components will be included in the pipeline.
  components = [
      train_gen,
      statistics_gen,
      schema_gen,
      transform,
      trainer,
      evaluator,
      pusher,
  ]

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      metadata_connection_config=tfx.orchestration.metadata
      .sqlite_metadata_connection_config(metadata_path),
      components=components
  )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Training on the MNIST dataset.")
    parser.add_argument(
        "-tr", "--dataset",
        type=str,
        help="Path where preprocessed train datasets are stored.",
        default=None
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        help="Pipeline name.",
        default=None
    )
    args = parser.parse_args()

    # Run DAG made by TFX
    tfx.orchestration.LocalDagRunner().run(
        _create_pipeline(
            pipeline_name=args.name,
            pipeline_root=os.path.join('pipelines', args.name),
            train_data_root=args.dataset,
            module_file='/media/linx/Secondary/C_T6/use-cases/mnist/tensorflow/components.py',
            serving_model_dir=os.path.join('serving_model', args.name),
            metadata_path=os.path.join('metadata', args.name, 'metadata.db')
        )
    )