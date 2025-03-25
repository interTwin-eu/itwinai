Pulsar Segmentation and Analysis for Radio-Astronomy (HTW Berlin)
===============================================================

The code is adapted from
`this repository <https://gitlab.com/ml-ppa/pulsarrfi_nn/-/tree/version_0.2/unet_semantic_segmentation?ref_type=heads>`_.
Please visit the original repository for more technical information on the code. 
This use-case features a sophisticated pipeline composed of few neural networks.

Scalability Metrics
-------------------
Here are some examples of the scalability metrics for this use case: 

Average Epoch Time Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This plot shows a comparison between the average time per epochs for each strategy
and number of nodes. 

.. image:: ../../use-cases/virgo/scalability-plots/absolute_scalability_plot.png

Relative Epoch Time Speedup
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This plot shows a comparison between the speedup between the different number of nodes
for each strategy. The speedup is calculated using the lowest number of nodes as a
baseline.

.. image:: ../../use-cases/virgo/scalability-plots/relative_scalability_plot.png

Communication vs Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This plot shows how much of the GPU time is spent doing computation compared to
communication between GPUs and nodes, for each strategy and number of nodes. The shaded
area is communication and the colored area is computation. They have all been
normalized so that the values are between 0 and 1.0. 

.. image:: ../../use-cases/virgo/scalability-plots/communication_plot.png

GPU Utilization
~~~~~~~~~~~~~~~
This plot shows how high the GPU utilization is for each strategy and number of nodes,
as a percentage from 0 to 100. This is the defined as how much of the time is spent
in computation mode vs not, and does not directly correlate to FLOPs. 

.. image:: ../../use-cases/virgo/scalability-plots/utilization_plot.png

Power Consumption
~~~~~~~~~~~~~~~~~
This plot shows the total energy consumption in watt-hours for the different strategies
and number of nodes. 

.. image:: ../../use-cases/virgo/scalability-plots/gpu_energy_plot.png
