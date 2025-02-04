Using the itwinai TorchTrainer Class
=========================================

The code used in this tutorial is adapted from
`this example <https://github.com/pytorch/examples/blob/main/mnist/main.py>`_.

The ``itwinai TorchTrainer`` class works as a wrapper that manages most aspects
of training. It facilitates distributed machine learning and allows for extensive
customization by subclassing and overriding the desired methods. 

You can find all the associated code in the 
`GitHub repository <https://github.com/interTwin-eu/itwinai/tree/main/tutorials/distributed-ml/torch-tutorial-2-trainer-class>`_.

Setting Up the Training Script
++++++++++++++++++++++++++++++

The following is an outline on how you can setup the training script: 

.. code-block:: python

    # Create dataset as usual
    train_dataset = ...

    # Create model as usual
    model = ...

    trainer = TorchTrainer(config={}, model=model, strategy="ddp")

    _, _, _, trained_model = trainer.execute(train_dataset, ...)

Launching Distributed Training
++++++++++++++++++++++++++++++

To launch the training across multiple workers, i.e. with multiple GPUs, potentially
across multiple nodes, you can use ``torchrun`` to allow the processes to communicate
between them. If you are on a system that uses SLURM, you can combine ``srun`` and
``torchrun`` to start the processes on different nodes as well. Here is an example
on how you could do this, assuming your code is in ``train.py``:

.. code-block:: bash

    srun --cpu-bind=none --ntasks-per-node=1 \
        bash -c "torchrun \
        --nnodes=2 \
        --nproc_per_node=4 \
        --rdzv_id=151152 \
        --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
        --rdzv_backend=c10d \
        --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
        python train.py"

Complete TorchTrainer Example
+++++++++++++++++++++++++++++

Below we have a complete example of how to use the TorchTrainer to train a model on
the MNIST dataset, which can be seen on Github
`here <https://github.com/interTwin-eu/itwinai/blob/main/tutorials/distributed-ml/torch-tutorial-2-trainer-class/train.py>`_. 
This can be run locally using 

.. code-block:: bash

  python train.py

or in a distributed manner as explained in the section above. If you wish to analyze
the resulting MLFlow logs, you can use the following command: 

.. code-block:: bash

   itwinai mlflow-ui --path mllogs/mlflow

.. note:: 
   You might have to change the port or the host, depending on which system you are on.
   If you are running this on a server and wish to port-forward the result to your local
   machine, then you have to change out the host using ``--host`` to ``0.0.0.0``. For
   more information on this, look for information on how to forward ports with SSH
   online. 

Here you can see the contents of ``train.py``: 

.. literalinclude:: ../../../tutorials/distributed-ml/torch-tutorial-2-trainer-class/train.py
   :language: python
   :linenos:
