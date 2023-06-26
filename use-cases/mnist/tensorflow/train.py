import os
import tensorflow_model_analysis as tfma
import argparse

from tfx import v1 as tfx


def create_pipeline(
        name: str,
        root: str,
        data_root: str,
        module_file: str,
        model_dir: str,
        metadata_path: str
) -> tfx.dsl.Pipeline:
    """
    Pipeline of TFX defining the procedure for working, training, evaluating with MNIST dataset
    """

    # Import data from TFRecords
    train_component = tfx.components.ImportExampleGen(input_base=data_root).with_id('train_get')

    # Statistics over data for visualization and further usage
    stat_component = tfx.components.StatisticsGen(examples=train_component.outputs['examples'])

    # Schema from statistics of the dataset
    schema_component = tfx.components.SchemaGen(statistics=stat_component.outputs['statistics'], infer_feature_shape=True)

    # Transforms the dataset
    transform_component = tfx.components.Transform(
      examples=train_component.outputs['examples'],
      schema=schema_component.outputs['schema'],
      module_file=module_file
    )

    # Training of the model based on user-defined training function in ${moodule_file}.run_fn(...)
    trainer_component = tfx.components.Trainer(
      module_file=module_file,
      examples=transform_component.outputs['transformed_examples'],
      transform_graph=transform_component.outputs['transform_graph'],
      schema=schema_component.outputs['schema'],
      train_args=tfx.proto.TrainArgs(num_steps=10000),
      eval_args=tfx.proto.EvalArgs(num_steps=5)
    )

    # Define config for the eval_component
    eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key='label')],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='SparseCategoricalAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0.75}
                      )
                  )
              )
          ])
      ])

    # Evaluate the models from the training
    eval_component = tfx.components.Evaluator(
      examples=train_component.outputs['examples'],
      model=trainer_component.outputs['model'],
      eval_config=eval_config
    )

    # Push the selected models from the Evaluator
    pusher_component = tfx.components.Pusher(
      model=trainer_component.outputs['model'],
      model_blessing=eval_component.outputs['blessing'],
      push_destination=tfx.proto.PushDestination(
          filesystem=tfx.proto.PushDestination.Filesystem(base_directory=model_dir)
      )
    )

    # Define component workflow
    components = [
      train_component,
      stat_component,
      schema_component,
      transform_component,
      trainer_component,
      eval_component,
      pusher_component,
    ]

    return tfx.dsl.Pipeline(
      pipeline_name=name,
      pipeline_root=root,
      metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
      components=components
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="TFX Pipeline for the training and evaluation model for the MNIST dataset"
    )
    parser.add_argument(
        "-tr", "--datadir",
        type=str,
        help="Path where preprocessed train datasets are stored",
        default=None
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        help="Pipeline name",
        default=None
    )
    parser.add_argument(
        "-c", "--component",
        type=str,
        help="Components filepath",
        default=None
    )
    args = parser.parse_args()

    # Run the pipeline using DAG
    tfx.orchestration.LocalDagRunner().run(
        create_pipeline(
            name=args.name,
            root=os.path.join('./use-cases/mnist/tensorflow/pipelines', args.name),
            data_root=args.datadir,
            module_file=args.component,
            model_dir=os.path.join('models', args.name),
            metadata_path=os.path.join('metadata', args.name, 'metadata.db')
        )
    )