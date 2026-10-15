Kubeflow pipeline
=================

Use a pipeline when training is one step among several and you want parameters and a run
history.

A Kubeflow pipeline is a graph of containers. itwinai fits as one container per step, and
because a pipeline step is just a command, you reuse the same ``config.yaml`` and the same
``itwinai exec-pipeline`` call as everywhere else.

A useful split is a cheap CPU step that prepares the dataset, followed by a GPU step that
trains. ``+pipe_steps`` selects part of the itwinai pipeline, so you do not need two config
files:

.. code-block:: python
   :caption: ``mnist_pipeline.py``

   # pip install "kfp==2.13.0" "kfp-kubernetes==1.5.0"
   from kfp import dsl, kubernetes

   IMAGE = "ghcr.io/intertwin-eu/itwinai:torch-slim-latest"
   WORKDIR = "/shared/<user>/mnist-usecase"
   GPU_MODEL = "NVIDIA-L40S"
   GPUS = 2


   @dsl.container_component
   def prepare_data(workdir: str):
       """Run only the data-getter step of the itwinai pipeline."""
       return dsl.ContainerSpec(
           image=IMAGE,
           command=["bash", "-c"],
           args=[
               f"cd {workdir} && export PYTHONPATH={workdir} && "
               "itwinai exec-pipeline --config-name config.yaml "
               "+pipe_key=training_pipeline +pipe_steps=[dataloading_step] "
               f"dataset_root={workdir}/data"
           ],
       )


   @dsl.container_component
   def train(workdir: str, epochs: int):
       """Run the full itwinai pipeline under torchrun, single node, GPUS processes."""
       return dsl.ContainerSpec(
           image=IMAGE,
           command=["bash", "-c"],
           args=[
               f"cd {workdir} && export PYTHONPATH={workdir} && "
               f"torchrun --nnodes=1 --nproc_per_node={GPUS} --no-python "
               "itwinai exec-pipeline --config-name config.yaml "
               "+pipe_key=training_pipeline strategy=ddp "
               f"epochs={epochs} dataset_root={workdir}/data "
               f"checkpoints_location={workdir}/checkpoints_kfp"
           ],
       )


   @dsl.pipeline(name="itwinai-mnist")
   def mnist_pipeline(epochs: int = 2):
       prep = prepare_data(workdir=WORKDIR)
       prep.set_caching_options(False)

       tr = train(workdir=WORKDIR, epochs=epochs)
       tr.after(prep)
       tr.set_caching_options(False)
       tr.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(GPUS)
       kubernetes.add_node_selector(tr, "nvidia.com/gpu.product", GPU_MODEL)

Compile and submit it:

.. code-block:: python

   from kfp import compiler
   from kfp.client import Client

   compiler.Compiler().compile(mnist_pipeline, "mnist_pipeline.yaml")

   client = Client()  # inside a notebook on the cluster
   client.create_run_from_pipeline_func(
       mnist_pipeline,
       namespace="<your-namespace>",
       experiment_name="itwinai-smoke",
       arguments={"epochs": 2},
   )

From a notebook on the cluster, ``Client()`` finds the pipeline service by itself.
