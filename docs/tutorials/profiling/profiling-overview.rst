Profiling Overview
==================

This is an overview over the different profiling methods used in ``itwinai``, as well as a
roadmap on when to use which profiler.

``itwinai`` Profilers — a quick intro
-------------------------------------

These are the different options for profiling your training with ``itwinai``:

* **Communication vs Computation vs Other**: Tries to approximate the time spent doing
  computation and communication to understand potential bottlenecks with the distribution
  across multiple GPUs.
* **GPU Energy Consumption and Utilization**: Measures how much energy is spent and the
  average utilization for the GPUs. 
* **Time per Epoch**: Measures how much time is spent per epoch to understand how well the
  training algorithm scales.
* **General Profiling with py-spy**: Measures how much time is spent in each function with
  statistical sampling to help you focus your optimization efforts on the right part of the
  code. 

The first three can be toggled with the following boolean flags in your configuration:

* ``torch_profiling``: Activate the PyTorch Profiler for communication vs computation vs other.
* ``store_torch_traces``: Store the traces from the PyTorch Profiler (requires 
  ``torch_profiling`` to be activated as well).
* ``measure_gpu_data``: Measure the GPU energy consumption and utilization.
* ``measure_epoch_time``: Measure the time per epoch.

As these flags are input parameters to the ``TorchTrainer``, make sure to place them under
this target, as shown in the following example:

.. code-block:: yaml

      ...
      training_step:
        _target_: itwinai.torch.trainer.TorchTrainer
        enable_torch_profiling: True
        store_torch_profiling_traces: True
        measure_gpu_data: True
        measure_epoch_time: True


For more information on how to activate the **py-spy** profiler, read the
:doc:`py-spy profiling guide <py-spy-profiling>`.


Selection Guide
===============

This section guides you in choosing the right profiler based on what you're trying to measure.
Some profilers are primarily intended for analyzing **scalability** across different training
setups, while others are best suited for **debugging general bottlenecks**.

Understanding Scalability
--------------------------

If you're running your code on multiple GPUs or nodes and want to evaluate how well it scales,
``itwinai`` provides several tools to help you break down where time is spent and how hardware
is used.

enable_torch_profiling
    Approximates time spent on communication vs computation to help identify scaling
    bottlenecks when running on multiple GPUs or nodes.

    Optionally enable ``store_torch_profiling_traces`` to view a timeline of activity in TensorBoard.

    .. warning::

       This measure is only a rough approximation, as it does not account for overlap in time.
       Also note that distributed training frameworks differ in their implementation, so
       comparisons across frameworks are not meaningful. Use this to compare how each strategy
       scales, not as an absolute measure of communication overhead.

measure_epoch_time
    Tracks the wall-clock time per epoch to evaluate how your training scales with more data or
    compute.

    This is a coarse but direct measure of scalability. The output can be plotted or compared
    across runs and configurations.

measure_gpu_data
    Monitors GPU energy consumption and utilization. Useful for assessing whether your GPUs are
    underutilized or your training is unnecessarily energy-intensive.

    Reports average utilization and total energy usage per GPU for the full training run.

Diagnosing Python-Side Bottlenecks
----------------------------------

py-spy
    External profiler that captures a statistical overview of where time is spent in your
    Python code.

    Particularly useful for spotting performance issues that are unrelated to scaling—such as
    slow Python loops, blocking calls, or I/O overhead. Best used when you're unsure where to
    begin optimizing.

    For more details, see the :doc:`py-spy profiling guide <py-spy-profiling>`.
