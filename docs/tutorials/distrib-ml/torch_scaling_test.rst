PyTorch scaling test
====================

.. include:: ../../../tutorials/distributed-ml/torch-scaling-test/README.md
   :parser: myst_parser.sphinx_


Plots of the scalability metrics
--------------------------------

We have the following scalability metrics available: 

- Absolute wall-clock time comparison
- Relative wall-clock time speedup
- Communication vs. Computation time
- GPU Utilization (%)
- Power Consumption (Watt)

Some examples of these scalability metrics on the Virgo use case with one, two and four
nodes respectively can be seen below: 

Absolute Wall-Clock Time Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../../tutorials/distributed-ml/torch-scaling-test/img/absolute-time.png

Relative Wall-Clock Time Speedup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../../tutorials/distributed-ml/torch-scaling-test/img/relative-speedup.png

Communication vs Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../../tutorials/distributed-ml/torch-scaling-test/img/comp-vs-comm.png

GPU Utilization
~~~~~~~~~~~~~~~

.. image:: ../../../tutorials/distributed-ml/torch-scaling-test/img/gpu-utilization.png

Power Consumption
~~~~~~~~~~~~~~~~~

.. image:: ../../../tutorials/distributed-ml/torch-scaling-test/img/energy-consumption.png
