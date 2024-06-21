Workflows
====================

Pipeline
===================


For simple workflows, itwinai defines a `Pipeline` object which is defined as an array or dictionary of ordered components.
The pipeline structure handles component connection by passing preceding components' output into following components' input. 

More advanced workflows for which the pipeline structure is too constraining require explicit connection of components


For each pipeline component, an `execute()` function is defined that provides a unified interface with every component as well as the whole pipeline.

Crucially, the input(s) for a component must be appropriate for that component.
Pipeline execution occurs in the order defined by the user. Thus, the output of a component must be suitable for the input of the following component.
The `ADAPT` component can be used to ensure this. It takes any number of inputs and will output some or all of those inputs in a user-specified order.


Below is given an overview of each component:


.. note::
    The `Pipeline` structure does not handle improper inputs for its components! 
    Each component expects predefined inputs which should be taken into account when constructing your Pipeline.
    The `Adapt` component can be used to ensure components receive the correct input if the preceding component's output is unsuited.