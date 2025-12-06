# Reproducing the results from the paper

Here we describe the procedure to reproduce the results shown in the plots of
the paper, concerning the physics use case targeting gravitational-wave analysis
at the [Virgo](https://www.virgo-gw.eu/) interferometer. More details on this
use case can be found [on this
page](https://www.intertwin.eu/article/thematic-module-glitchflow).

Below you will find instructions both for reproducing the experiments on an HPC
system similar to [JUWELS
Booster](https://apps.fz-juelich.de/jsc/hps/juwels/index.html), the one used to
run the experiments shown on the paper, and for *single-host* environments
(e.g., laptop, VM, single HPC node) that are usually accessed via interactive
shells.

We assume that on HPC you have access to a SLURM scheduler, whereas on a
single-host setting you have access to Docker or equivalent (e.g., podman).

## Environment

Below some instruction on how to reuse the same dependencies and itwinai
version.

The results in the paper were obtained with itwinai `v0.3.4`, and the same
dependencies can be found by using the following OCI container:
`ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments`

HPC systems generally prefer Apptainer or Singularity over Docker. If on an HPC,
first download the container:

```bash
singularity pull itwinai.sif docker://ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments
```

## Machine learning workflow

In this case, for our experiments we will use a small synthetic dataset
generated on the fly.  The original data collected at the Virgo detector is not
open.

The machine learning workflow is described in the `training_pipeline_small` in
the `config.yaml` configuration file. This pipeline describes add the steps in
the ML workflow, including dataset generation, train-validation split,
pre-processing, and training.

Alternatively, you could generate a larger dataset and run a more
compute-intensive workload using the `training_pipeline` pipeline in
`config.yaml`. However, this is not covered in this guide and the interested
reader is referred to the `README-use-case.md` file.

## Dataset Generation

First we need to generate a synthetic training dataset. For the sake of
simplicity, we will generate a smaller one compared to the one used in the
experiments reported in the paper. Therefore, the results may not be identical.
In particular, do not pay too much attention to the loss curves as the model
may not be able to learn well on such this synthetic dataset.

On an single-host setup, generate a sample training dataset by executing only
the first step in the pipeline. Notice that the local `$PWD/data` directory is
mounted on `/data` to store the data on a persistent location. Remember to mount
that every time you want to use the generated dataset.

```bash
docker run --rm -v "$PWD/data":/data -v "$PWD":/experiments --user $UID \
    ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments \
    itwinai exec-pipeline +pipe_key=training_pipeline_small +pipe_steps=[0] 
```

On an HPC system you can run the same command on a login node with small
adaptations, or submit a SLURM job in the following way:

```bash
sbatch synthetic-data-gen/data_generation_small.sh
```

At the end of dataset generation, you should see some pickle files:

```bash
$ ls data/*.pkl
Image_dataset_synthetic_64x64.pkl  TimeSeries_dataset_synthetic_aux.pkl  TimeSeries_dataset_synthetic_main.pkl
```

To force the re-generation of the dataset, remove them.

## Training and scalability metrics

To begin with, please refer to the [scalability report
guide](https://itwinai.readthedocs.io/v0.3.4/how-it-works/scalability-report/scalability_report.html)
and to the [introduction to profiling with
itwinai](https://itwinai.readthedocs.io/v0.3.4/tutorials/profiling/profiling-overview.html).

To collect scalability metrics set the following fields in the trainer
configuration:

```yaml
measure_gpu_data: False
enable_torch_profiling: True
measure_epoch_time: True
```

These are already the default values in the provided configuration file.

In this guide we will not discuss the collection of GPU metrics, so
`measure_gpu_data` is set to `False`.

Before running a scalability test, on a single-host environment you can check
that training works by executing this command as a dry run:

```bash
docker run --rm -v "$PWD/data":/data -v "$PWD":/experiments --user $UID \
    ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments \
    itwinai exec-pipeline +pipe_key=training_pipeline_small
```

This will run training in a non-distributed way using the synthetic dataset
generated before. The results will be stored in the current working directory.

On an HPC system you can adapt the command above for Singularity.

### Scalability experiment

Here we run a set of experiments changing the number of workers used for
distributed data-parallel ML training.

On a single-host setting run:

```bash
bash scalability/scalability_local.sh
```

This script will launch multiple training runs with different number of workers each.

On HPC...TODO

### Visualize the scalability metrics

Once all the trainings are completed, generate the **scalability report**.

On a single-host machine you can do the following:

```bash
docker run --rm -v "$PWD/data":/data -v "$PWD":/experiments --user $UID \
    ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments \
    itwinai generate-scalability-report --experiment-name virgo-small
```

You can see the generated plots in the `plots` directory. Please consider that
the plots may be different from the ones shown on the paper both because you may
be using different hardware and because the dataset is different. Also, consider
that when running scalability analysis on small datasets there may be quite some
noise in the plots.

You can also inspect the training metrics collected during training with MLflow:

```bash
docker run --rm -v "$PWD/data":/data -v "$PWD":/experiments --user $UID -p 5000:5000\
    ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments \
    mlflow ui --host 0.0.0.0 --backend-store-uri mllogs/mlflow
```

On the login node of an HPC you can run the commands above, adapting the Docker
syntax to Apptainer/Singularity.
