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

You can see example plots of these in the 
:doc:`Virgo documentation <../../use-cases/virgo_doc>` or the 
:doc:`EURAC documentation <../../use-cases/eurac_doc>`.

Additionally, we ran a larger scalability test with this tutorial on the full ImageNet
dataset with the older script. This only shows the relative speedup and can be seen here:

.. image:: ../../../tutorials/distributed-ml/torch-scaling-test/img/report.png

