Profiling Roadmap
=================

This is an overview over the different profiling methods used in ``itwinai``, as well as a
roadmap on when to use which profiler.

Profiler Overview
-----------------

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
        torch_profiling: True
        store_torch_traces: True
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

**Tracking Communication Overhead**

When scaling to more GPUs, increased communication costs can slow things down. To understand
how much time is lost to communication (vs. actual computation):

→ **Enable**: ``torch_profiling``  

Optionally enable ``store_torch_traces`` to view a timeline in TensorBoard.

.. warning::

   This measure is only a rough approximation as it does not account for overlap in time.
   Additionally, since the different frameworks for distributed learning vary in their
   implementation, a direct comparison between frameworks cannot be made. Use this to compare
   how each strategy scales, but not as an overall measure of communication overhead. 

**Comparing Wall-Clock Time Across Runs**

To get a coarse but direct measure of scalability, track how long each epoch takes as you scale
up data or compute:

→ **Enable**: ``measure_epoch_time``  

Gives per-epoch timing that can be plotted or compared across configurations.

**Analyzing GPU Efficiency**

To measure the energy cost of your training, or check whether your GPUs are sitting idle too
much:

→ **Enable**: ``measure_gpu_data``  

This provides average utilization and energy use for each GPU during training.

Diagnosing Python-Side Bottlenecks
----------------------------------

If training is slow even on a single GPU or small setup, you may have issues unrelated to
scaling—such as inefficient loops, slow I/O, or blocking calls. For more information about
our **py-spy** integration, read the :doc:`py-spy profiling guide <py-spy-profiling>`.

→ **Use**: py-spy 

This runs externally and gives you a statistical overview of where time is spent in your Python
code. Best used when you don’t know where to start optimizing.

