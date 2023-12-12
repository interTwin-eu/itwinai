"""
The most simple workflow that you can write is a sequential pipeline of steps,
where the outputs of a component are fed as input to the following component,
employing a scikit-learn-like Pipeline.

This allows to export the Pipeline form Python code to configuration file, to
persist both parameters and workflow structure. Exporting to configuration file
assumes that each component class resides in a separate python file, so that
the pipeline configuration is agnostic from the current python script.

Once the Pipeline has been exported to configuration file (YAML), it can
be executed directly from CLI:

>>> itwinai exec-pipeline --config my-pipeline.yaml --override nested.key=42

The itwinai CLI allows for dynamic override of configuration fields, by means
of nested key notation. Also list indices are supported:

>>> itwinai exec-pipeline --config my-pipe.yaml --override nested.list.2.0=42

"""
from itwinai.pipeline import Pipeline
from itwinai.parser import ConfigParser

from basic_components import MyDataGetter, MyDatasetSplitter, MyTrainer

pipeline = Pipeline([
    MyDataGetter(data_size=100),
    MyDatasetSplitter(
        train_proportion=.5, validation_proportion=.25, test_proportion=0.25
    ),
    MyTrainer()
])

# Run pipeline
_, _, _, trained_model = pipeline.execute()
print("Trained model: ", trained_model)
print("\n" + "="*50 + "\n")

# Serialize pipeline to YAML
pipeline.to_yaml("basic_pipeline_example.yaml", "pipeline")

# Below, we show how to run a pre-existing pipeline stored as
# a configuration file, with the possibility of dynamically
# override some fields

# Load pipeline from saved YAML (dynamic serialization)
parser = ConfigParser(
    config="basic_pipeline_example.yaml",
    override_keys={
        "pipeline.init_args.steps.0.init_args.data_size": 200
    }
)
pipeline = parser.parse_pipeline()
print(f"MyDataGetter's data_size is now: {pipeline.steps[0].data_size}\n")

# Run parsed pipeline, with new data_size for MyDataGetter
_, _, _, trained_model = pipeline.execute()
print("Trained model (2): ", trained_model)

# Save new pipeline we YAML file
pipeline.to_yaml("basic_pipeline_example_v2.yaml", "pipeline")
