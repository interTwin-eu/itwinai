Tropical Cyclones Detection
===========================

The code is adapted from the CMCC use case's
`repository <https://github.com/CMCC-Foundation/ml-tropical-cyclones-detection>`_.

To know more on the interTwin tropical cyclones detection use case and its DT, please visit the published deliverables,
`D4.1 <https://zenodo.org/records/10417135>`_, 
`D7.1 <https://zenodo.org/records/10417158>`_ and `D7.3 <https://zenodo.org/records/10224252>`_.

Setup env
+++++++++

.. code-block:: bash

    # After activating the environment
    pip install -r requirements.txt


Dataset
+++++++

If the automatic download from python does not work, try from the command line from
within the virtual environment:

.. code-block:: bash

    gdown https://drive.google.com/drive/folders/1TnmujO4T-8_j4bCxqNe5HEw9njJIIBQD -O data/tmp_data/trainval --folder


For more info visit the `gdown <https://github.com/wkentaro/gdown>`_ repository.

Training
++++++++

Launch training:

.. code-block:: bash

    # # ONLY IF tensorflow>=2.16
    # export TF_USE_LEGACY_KERAS=1

    source ../../.venv-tf/bin/activate
    python train.py -p pipeline.yaml 


On JSC, the dataset is pre-downloaded and you can use the following command:

.. code-block:: bash

    # # ONLY IF tensorflow>=2.16
    # export TF_USE_LEGACY_KERAS=1

    source ../../envAItf_hdfml/bin/activate
    python train.py -p pipeline.yaml --data_path /p/project/intertwin/smalldata/cmcc

    # Launch a job with SLURM
    sbatch startscript.sh

pipeline.yaml
+++++++++++++

This YAML file defines the pipeline configuration for the CMCC use case.

.. literalinclude:: ../use-cases/cyclones/pipeline.yaml
   :language: yaml

train.py
++++++++++
.. literalinclude:: ../use-cases/cyclones/train.py
   :language: python

dataloader.py
+++++++++++++

.. literalinclude:: ../use-cases/cyclones/dataloader.py
   :language: python

trainer.py
++++++++++
.. literalinclude:: ../use-cases/cyclones/trainer.py
   :language: python

startscript
+++++++++++

.. literalinclude:: ../use-cases/cyclones/startscript.sh
   :language: bash

cyclones_vgg.py
+++++++++++++++

.. literalinclude:: ../use-cases/cyclones/cyclones_vgg.py
   :language: python
