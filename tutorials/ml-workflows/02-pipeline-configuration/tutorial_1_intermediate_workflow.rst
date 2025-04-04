.. _using_configuration_files:

=======================================
Using Configuration Files in itwinai
=======================================

Pipeline and configuration files
================================

**Author(s)**: Anna Lappe (CERN), Matteo Bunino (CERN)

In the previous tutorial, we introduced how to create new components and assemble them into a 
**Pipeline** for a simplified workflow execution. The **Pipeline** executes components in the 
order they are defined, assuming that each component's outputs will fit as inputs to the next one.

Sometimes, you might want to define your pipeline in a configuration **YAML** file instead. 
This allows for:

- Easier modification and reuse of pipeline definitions
- Clear separation between code and configuration
- Dynamic overrides of parameters at runtime

Example Configuration File
==========================

.. code-block:: yaml

    training_pipeline:
      _target_: itwinai.pipeline.Pipeline
      steps:
        my-getter:
          _target_: basic_components.MyDataGetter
          data_size: 200
        my-splitter:
          _target_: basic_components.MyDatasetSplitter
          train_proportion: 0.5
          validation_proportion: 0.25
          test_proportion: 0.25
        my-trainer: 
          _target_: basic_components.MyTrainer

itwinai uses `Hydra <https://hydra.cc>`_ to parse configuration files and instantiate 
pipelines dynamically. There are two ways you can use your configuration file to run a pipeline: 
from the command-line interface (CLI) or from within your code.

Parsing Pipelines from CLI
==========================

You can execute a pipeline from a configuration file (which by default is called ``config.yaml``)
with the CLI using:

.. code-block:: bash

    itwinai exec-pipeline

This command loads the configuration file and executes the defined pipeline. 
You can customize execution using the following options:

1. Setting a ``--config-path`` and ``--config-name``
-----------------------------------------------------

By default, the parser will look for a file called ``config.yaml`` inside your current working 
directory. If you want to change this, set the path to your configuration file with the 
``--config-path`` option. This can be either absolute or relative to your current working 
directory and should point to the directory in which your configuration file is located. 
If your configuration file has a different name (not ``config.yaml``), you may specify this with the ``--config-name`` flag:

.. code-block:: bash

    itwinai exec-pipeline --config-path path/to/dir --config-name my-config-file

Note that we omit the extension ``.yaml``. This is intentional, as hydra expects only the stem 
of the filename.

2. Selecting a pipeline by name (``+pipe_key``)
-----------------------------------------------

A configuration file can contain multiple pipelines. The default key that the parser will look 
for is ``training_pipeline``. Use the ``pipe_key`` argument to overwrite this default and 
specify which pipeline to execute:

.. code-block:: bash

    itwinai exec-pipeline +pipe_key=another_training_pipeline

3. Selecting Steps to Run (``+pipe_steps``)
-------------------------------------------

If you only want to run a subset of specific steps of the pipeline, use ``pipe_steps``:

.. code-block:: bash

    itwinai exec-pipeline +pipe_steps=[my-splitter, my-trainer]

This will execute only the ``MyDatasetSplitter`` and ``MyTrainer`` steps of the pipeline. You can also 
give ``pipe_steps`` as a list of indices, if your configuration file defines your steps in list format.

4. Dynamically overriding configuration fields
----------------------------------------------

You can override any parameter in the configuration file directly from the command line:

.. code-block:: bash

    itwinai exec-pipeline +training_pipeline.steps.my-getter.data_size=500

This modifies the ``data_size`` parameter inside the pipeline configuration. You can also override 
fields if your pipeline steps are defined in the form of a list in your configuration file.
In this case, you give the step's index instead of its name, for example

.. code-block:: bash

    itwinai exec-pipeline +training_pipeline.steps.0.data_size=500


Advanced Functionality with Hydra
=================================

Since this implementation is based on **Hydra**, you can use all of Hydra’s command-line arguments, 
such as for multi-run execution, merging configuration files, and debugging. For more details, 
refer to the `Hydra documentation <https://hydra.cc/docs/advanced/hydra-command-line-flags/>`_.

.. note::

    If your pipeline execution fails and you need detailed error messages, 
    we recommend that you set the following environment variable before running the pipeline:

    .. code-block:: bash

        export HYDRA_FULL_ERROR=1

    This will give you more verbose error messages including the full stack trace given by Hydra.
    
    If you do not want the variable to persist, i.e. you only want to run your command with the
    the detailed error message once, you can also run it such that the environment variable 
    ``HYDRA_FULL_ERROR`` will not persist and reset after your command has been executed:

    .. code-block:: bash

        HYDRA_FULL_ERROR=1 itwinai exec-pipeline


Parsing Pipelines from Python
=============================

In some cases, you may want to parse and execute a pipeline from a configuration file from within 
your Python code. You can do this by running:

.. code-block:: python

    from hydra import compose, initialize
    from itwinai import exec_pipeline_with_compose

    # Here, we show how to run a pre-existing pipeline stored as
    # a configuration file from within Python code, with the possibility of dynamically
    # overriding some fields

    # Load pipeline from saved YAML (dynamic deserialization)
    with initialize():
        cfg = compose(
            config_name="my-config.yaml",
            overrides=[
                "pipeline.steps.0.data_size=400",
            ],
        )
        exec_pipeline_with_compose(cfg)

Reproducibility
===============

Each execution logs the pipeline configuration under the ``outputs/`` directory. This ensures 
reproducibility by recording the exact parameters used for execution.

