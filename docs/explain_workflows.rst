Workflows
====================

Pipeline:
------------------

For simple workflows, itwinai defines a `Pipeline`, defined as an array or dictionary of ordered components.
Much like a python notebook, the Pipeline sequentially runs through the user-defined components inside of it, ensuring code legibility and organisation.
The pipeline execution also avoids writing to memory when moving to the next component, ensuring efficient workflows.
The pipeline structure handles component connection by passing preceding components' output into following components' input whilst staying in-memory, similarly to scikit-learn.
Bypassing the need to write to disk allows for more efficient and reproducible (?) workflows.
This also implies that the Pipeline structure only handles sequential workflows; more advanced non-sequential workflows cannot take advantage of `Pipeline`.

the `Pipeline` is fully represented by its **configuration file**. 
This files contains all the parameters and structure variables required to uniquely identify a `Pipeline`.

Crucially, the input(s) for a component must be appropriate for that component.
Pipeline execution occurs in the order defined by the user. Thus, the output of a component must be suitable for the input of the following component.
The `ADAPT` component can be used to ensure this. It takes any number of inputs and will output some or all of those inputs in a user-specified order.


Below is given an overview of each component:

Components:
------------------
Components are defined as discrete steps in a `Pipeline`.
For each pipeline component, an `execute()` function is defined that provides a unified interface with every component as well as the whole pipeline.

GET
^^^^^^^^^^^^^^
.. image:: figures/comp_Get.png
    :scale: 12%


SPLIT
^^^^^^^^^^^^^
.. image:: figures/comp_Split.png
    :scale: 12%

PROCESS
^^^^^^^^^^^^^^^^
.. image:: figures/comp_Proc.png
    :scale: 12%


TRAIN
^^^^^^^^^^^^^^^^
.. image:: figures/comp_Train.png
    :scale: 12%

ADAPT
^^^^^^^^^^^^^^
.. image:: figures/comp_Adapt.png
    :scale: 12%

PREDICT
^^^^^^^^^^^^
.. image:: figures/comp_Predict.png
    :scale: 12%


.. note::
    The `Pipeline` structure does not handle improper inputs for its components! 
    Each component expects predefined inputs which should be taken into account when constructing your Pipeline.
    The `Adapt` component can be used to ensure components receive the correct input if the preceding component's output is unsuited.
    For example, `Split` returns three data arrays whereas `Save` only takes one input argument.
    To save after a split, `Adapt` can be used to select the element to be saved.



.. image:: figures/simple_pipeline.png
    :alt: Diagram of a simple pipeline structure
    :align: center

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Contents

    notebooks/example