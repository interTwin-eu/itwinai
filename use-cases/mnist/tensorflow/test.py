import apache_beam as beam
import tensorflow_model_analysis as tfma
import argparse
import glob
import os

from tfx_bsl.public import tfxio
from google.protobuf import text_format


def run_pipeline(
        eval_config,
        model_path,
        dataset_path,
        output_path
):
    """
    Evaluation pipeline for the testing of the trained models
    """
    model = tfma.default_eval_shared_model(
        eval_saved_model_path=model_path,
        eval_config=eval_config
    )
    dataset = tfxio.TFExampleRecord(
        file_pattern=dataset_path,
        raw_record_column_name=tfma.ARROW_INPUT_COLUMN
    )

    # Run Evaluation.
    with beam.Pipeline() as pipeline:
        _ = (
            pipeline
            | 'ReadData' >> dataset.BeamSource()
            | 'EvalModel' >> tfma.ExtractEvaluateAndWriteResults(
               eval_shared_model=model,
               eval_config=eval_config,
               output_path=output_path
        )
    )

    result = tfma.load_eval_result(output_path=output_path)
    print(result)

    # Visualization works only in Jupyter
    # tfma.view.render_slicing_metrics(result)


if __name__ == '__main__':
    # Define the evaluation config file
    eval_config = text_format.Parse("""
      ## Model information
      model_specs {
        # For keras and serving models, you need to add a `label_key`.
        label_key: "label"
      }
    
      ## This post-training metric information is merged with any built-in
      ## metrics from training.
      metrics_specs {
        metrics { class_name: "ExampleCount" }
        metrics { class_name: "MeanAbsoluteError" }
        metrics { class_name: "MeanSquaredError" }
        metrics { class_name: "MeanPrediction" }
      }
    
      slicing_specs {}
    """, tfma.EvalConfig())

    parser = argparse.ArgumentParser(
        description="Testing of the MNIST dataset.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output path where the evaluation results are stored.",
        default=None
    )
    parser.add_argument(
        "-m", "--models",
        type=str,
        help="Input of the model to use for the evaluation.",
        default=None
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        help="Dataset path to be used for testing.",
        default=None
    )
    args = parser.parse_args()
    for it in os.scandir(args.models):
        if it.is_dir():
            # Append format of the saving TFX models folder
            model_path = os.path.join(it.path, 'Format-Serving')
            run_pipeline(
                eval_config=eval_config,
                model_path=model_path,
                dataset_path=args.dataset,
                output_path=args.output
            )
