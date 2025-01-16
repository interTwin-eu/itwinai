# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""TensorFlow distributed strategies."""

import os
from typing import Tuple

import tensorflow as tf
import tensorflow.distribute as dist


def get_strategy() -> Tuple[tf.distribute.Strategy, int]:
    """Strategy for distributed TensorFlow training. It will automatically
    detect if you are running in a multi-node environment, returning the
    correct TensorFlow distributed strategy for data parallel distributed
    training.

    Returns:
        Tuple[tf.distribute.Strategy, int]: strategy and number of
        `parallel workers`_.

        .. _parallel workers:
            https://stackoverflow.com/questions/66005641/why-we-are-using-strategy-num-replicas-in-sync.
    """

    slurm_jobid = os.environ.get("SLURM_JOB_ID")
    slurm_nnodes = int(os.environ.get("SLURM_NNODES", 0))
    if not slurm_jobid or slurm_nnodes < 2:
        # Single-node environment
        print("Not in SLURM env! Assuming that you are running on a single node")
        mirrored_strategy = dist.MirroredStrategy()
        return mirrored_strategy, mirrored_strategy.num_replicas_in_sync

    # Multi-node environment in SLURM
    cluster_resolver = dist.cluster_resolver.SlurmClusterResolver(port_base=12345)
    implementation = dist.experimental.CommunicationImplementation.NCCL
    communication_options = dist.experimental.CommunicationOptions(
        implementation=implementation
    )

    # declare distribution strategy
    tf_dist_strategy = dist.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver, communication_options=communication_options
    )

    # number of workers
    n_workers = int(os.environ["SLURM_NTASKS"])
    # list of devices per worker
    devices = tf.config.experimental.list_physical_devices("GPU")
    # number of devices per worker
    n_gpus_per_worker = len(devices)
    # total number of GPUs
    n_gpus = n_workers * n_gpus_per_worker

    # get total number of detected GPUs
    print("Number of detected devices: {}".format(n_gpus))

    # get total number of workers
    print("Number of devices: {}".format(tf_dist_strategy.num_replicas_in_sync))

    return tf_dist_strategy, tf_dist_strategy.num_replicas_in_sync
