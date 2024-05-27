Tensorflow ImageNet example
===========================

Tutorial: distributed strategies for Tensorflow
-----------------------------------------------

In this tutorial we show how to use Tensorflow ``MultiWorkerMirroredStrategy``.
Note that the environment is tested on the HDFML system at JSC.
For other systems, the module versions might need change accordingly.
Other strategies will be updated here.

First, from the root of this repository, build the environment containing
Tensorflow. You can *try* with:

.. code-block:: bash

    # Creates a Python venv called envAItf_hdfml
    make tf-gpu-jsc

If you want to distribute the code in ``train.py``, run from terminal:

.. code-block:: bash

    sbatch tfmirrored_slurm.sh


train.py
++++++++

.. literalinclude:: ../tutorials/distributed-ml/tf-tutorial-1-imagenet/train.py
   :language: python


tfmirrored_slurm.sh
+++++++++++++++++++

.. literalinclude:: ../tutorials/distributed-ml/tf-tutorial-1-imagenet/tfmirrored_slurm.sh
   :language: bash

