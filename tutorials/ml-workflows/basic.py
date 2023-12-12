"""
The most simple workflow that you can write is a sequential pipeline of steps,
where the outputs of a component are fed as input to the following component,
employing a scikit-learn-like Pipeline.

This allows to export the Pipeline form Python code to configuration file, to
persist both parameters and workflow structure. Exporting to configuration file
assumes that each component class resides in a separate python file, so that
the pipeline configuration is agnostic from the current python script.

Once the Pipeline has been exported to configuration file
"""
