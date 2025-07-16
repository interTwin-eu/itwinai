Scalability Report
==================

The ``itwinai`` Scalability Report provides insights into how well your model's
performance scales when training across multiple GPUs and nodes. It can be used to find
bottlenecks or bugs as you distribute your workload, as well as help you monitor your
model's sustainability. The main goals are as follows:

- Understand how well your model scales with regards to the given metrics
- Discover which distributed strategy works best for your pipeline

The Scalability Report includes the following metrics: 

- Average time per epoch
- Relative speedup of time per epoch
- GPU Utilization (0-100%)
- GPU Power Consumption (watt-hours)
- Computation vs. other

For more information about profiling with ``itwinai``, you can have a look at the
:doc:`profiling overview <../../tutorials/profiling/profiling-overview>`.

.. note:: 

   To use the Scalability Report functionality of ``itwinai``, you need to use a
   Trainer object that inherits the ``itwinai`` ``TorchTrainer``. 

Generating the Data
-------------------

To generate the data, you have to train a model with the various profilers enabled,
which can be toggled using the following flags in your training configuration:

.. code-block:: yaml

   steps:
    - ...
    - _target_: <your-trainer-class>
      measure_gpu_data: True  # Measures GPU utilization and power consumption
      enable_torch_profiling: True  # Enable profiling for comparing comm. time vs. other
      store_torch_profiling_traces: True # Store the traces from the profiling (requires setting enable_torch_profiler to True)
      measure_epoch_time: True  # Measures avg. epoch time and rel. speedup
      ...

The epoch time is measured using the `EpochTimeTracker`, while the remaining metrics
are measured using the following decorators:

- **PyTorch Profiler**: This profiler measures the time spent in computation vs. other in your
  distributed machine learning. This is done by comparing the time spent in PyTorch's ATen
  library with the rest of the time. It uses the PyTorch Profiler to retrieve this
  information and is enabled using the ``enable_torch_profiling`` flag.
- **GPU Data Profiler**: This profiler measures the GPU utilization and the total power
  consumption of the training. This is done by probing the GPU at a pre-defined interval
  and retrieving the needed data. The data from the GPU data profiler is saved to MLFlow and
  therefore not a part of the `generate-scalability-report` command.

If you overwrite the ``TorchTrainer``'s ``train()`` method, then the decorators need to
be placed above your overwritten ``train()`` method as in the following example:

.. code-block:: python
   
  from itwinai.torch.monitoring.monitoring import measure_gpu_utilization
  from itwinai.torch.profiling.profiler import profile_torch_trainer

  class MyTrainer(TorchTrainer):
     ...

     @profile_torch_trainer
     @measure_gpu_utilization
     def train(self, ...):
        # Your train method here

If your profilers are enabled in the configuration—and if applicable, your decorators have
been appropriately positioned above your ``train()`` method—then this will create a
directory named ``scalability-metrics`` in the current working directory, under which
three subdirectories will be created: 

- ``epoch-time``: The wall-clock time data from the ``EpochTimeTracker``
- ``computation-data``: The computation overhead data

The GPU data will be written to MLFlow. 

Generating the Report
---------------------

You can generate the report using the following command: 

.. code-block:: bash

   itwinai generate-scalability-report

This command takes in some extra arguments that can be viewed with the ``--help`` flag:

.. code-block:: bash

   itwinai generate-scalability-report --help

When running this command by default, it will look in your ``scalability-metrics``
directory and look for the subdirectories listed above. Only the reports relevant to
the subdirectories that are present will be created, while missing subdirectories will only
result in a warning.

Example Results
---------------

The following will show some example results from the Virgo use case. Note that this
was run once and might therefore have some slightly distorted results due to random
noise in the training, such as some nodes performing better than others. To mitigate
this, one can run more analyses and aggregate the results.

The report will result in a table of scalability results, printed in the console, as
well as plots showing the same results visually. An example of the resulting console
output can be seen here # TODO: This needs to be updated

.. code-block::

    ######## Epoch Time Report ########
         name  nodes avg_epoch_time
    deepspeed      1        59.01 s
    deepspeed      2        31.37 s
    deepspeed      4        17.86 s
    deepspeed      8         9.48 s
      horovod      1        59.77 s
      horovod      2        34.91 s
      horovod      4        21.95 s
      horovod      8        16.75 s
    torch-ddp      1        72.92 s
    torch-ddp      2        48.62 s
    torch-ddp      4        19.26 s
    torch-ddp      8        10.30 s
    Saved absolute average time plot at '/Users/jarl/cern/cern_projects/itwinai/plots/absolute_epoch_time.png'.
    Saved relative average time plot at '/Users/jarl/cern/cern_projects/itwinai/plots/relative_epoch_time_speedup.png'.

    ######## Computation Data Report ########
     strategy  num_gpus computation_fraction
    deepspeed         4              00.09 %
    deepspeed         8              00.08 %
    deepspeed        16              00.08 %
    deepspeed        32              00.09 %
      horovod         4              00.77 %
      horovod         8              00.24 %
      horovod        16              00.21 %
      horovod        32              00.58 %
    torch-ddp         4              00.50 %
    torch-ddp         8              00.79 %
    torch-ddp        16              00.03 %
    torch-ddp        32              00.62 %
    Saved computation fraction plot at '/Users/jarl/cern/cern_projects/itwinai/plots/computation_fraction_plot.png'.

In this case, data was collected for 4, 8, 16 and 32 GPUs for the ``DeepSpeed``, ``Horovod``
and ``PyTorch DDP`` strategies. The associated plots can be seen below: 

Average Epoch Time Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This plot shows a comparison between the average time per epochs for each strategy
and number of nodes. 

.. image:: ./images/absolute_epoch_time.png

Relative Epoch Time Speedup
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This plot shows a comparison between the speedup between the different number of nodes
for each strategy. The speedup is calculated using the lowest number of nodes as a
baseline.

.. image:: ./images/relative_epoch_time_speedup.png

Computation vs other
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This plot shows how much of the GPU time is spent doing computation compared to
communication between GPUs and nodes, for each strategy and number of nodes. The shaded
area is communication and the colored area is computation. They have all been
normalized so that the values are between 0 and 1.0. 

.. image:: ./images/computation_fraction_plot.png
