Virgo
=====

The code is adapted from
`this notebook <https://github.com/interTwin-eu/DT-Virgo-notebooks/blob/main/WP_4_4/interTwin_wp_4.4_synthetic_data.ipynb>`_
available on the Virgo use case's `repository <https://github.com/interTwin-eu/DT-Virgo-notebooks>`_.

To know more on the interTwin Virgo Noise detector use case and its DT, please visit the published deliverables,
`D4.2 <https://zenodo.org/records/10417138>`_, 
`D7.2 <https://zenodo.org/records/10417161>`_ and `D7.4 <https://zenodo.org/records/10224277>`_.

Installation
++++++++++++

Before continuing, install the required libraries in the pre-existing itwinai environment.

.. code-block:: bash

    pip install -r requirements.txt


Training
++++++++

You can run the whole pipeline in one shot, including dataset generation, or you can
execute it from the second step (after the synthetic dataset have been generated).

.. code-block:: bash

    itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline

    # Run from the second step (use python-like slicing syntax).
    # In this case, the dataset is loaded from "data/Image_dataset_synthetic_64x64.pkl"
    itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps 1:


Launch distributed training with SLURM using the dedicated ``slurm.sh`` job script:

.. code-block:: bash

    # Distributed training with torch DistributedDataParallel
    PYTHON_VENV="../../envAI_hdfml"
    DIST_MODE="ddp"
    RUN_NAME="ddp-virgo"
    TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline --config config.yaml --steps 1: --pipe-key training_pipeline -o strategy=ddp"
    sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
        slurm.sh


...and check the results in ``job.out`` and ``job.err`` log files.

To understand how to use all the distributed strategies supported by ``itwinai``,
check the content of ``runall.sh``:

.. code-block:: bash

    bash runall.sh


> [!WARNING]
> The file ``train.py`` is not to be considered the suggested way to launch training,
> as it is deprecated and is there to testify an intermediate integration step
> of the use case into ``itwinai``.

When using MLFLow logger, you can visualize the logs in from the MLFlow UI:

.. code-block:: bash

    mlflow ui --backend-store-uri mllogs/mlflow

    # In background 
    mlflow ui --backend-store-uri mllogs/mlflow > /dev/null 2>&1 &


config.yaml
+++++++++++
.. literalinclude:: ../use-cases/virgo/config.yaml
   :language: yaml


data.py
+++++++

.. literalinclude:: ../use-cases/virgo/data.py
   :language: python


runall.sh
+++++++++

.. literalinclude:: ../use-cases/virgo/runall.sh
   :language: bash


slurm.sh
++++++++

.. literalinclude:: ../use-cases/virgo/slurm.sh
   :language: bash


trainer.py
++++++++++
.. literalinclude:: ../use-cases/virgo/trainer.py
   :language: python


