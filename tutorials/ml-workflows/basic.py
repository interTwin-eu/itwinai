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
from itwinai