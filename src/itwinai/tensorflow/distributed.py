import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tensorflow.distribute as dist


def get_strategy():
    """Strategy for distributed TensorFlow training"""
    if not os.environ.get('SLURM_JOB_ID'):
        # TODO: improve
        print('not in SLURM env!')
        tf_dist_strategy = dist.MirroredStrategy()
        return tf_dist_strategy, tf_dist_strategy.num_replicas_in_sync
    cluster_resolver = dist.cluster_resolver.SlurmClusterResolver(
        port_base=12345)
    implementation = dist.experimental.CommunicationImplementation.NCCL
    communication_options = dist.experimental.CommunicationOptions(
        implementation=implementation)

    # declare distribution strategy
    tf_dist_strategy = dist.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver,
        communication_options=communication_options
    )

    # number of workers
    n_workers = int(os.environ['SLURM_NTASKS'])
    # list of devices per worker
    devices = tf.config.experimental.list_physical_devices('GPU')
    # number of devices per worker
    n_gpus_per_worker = len(devices)
    # total number of GPUs
    n_gpus = n_workers * n_gpus_per_worker

    # get total number of detected GPUs
    print("Number of detected devices: {}".format(
        n_gpus))

    # get total number of workers
    print("Number of devices: {}".format(
        tf_dist_strategy.num_replicas_in_sync))

    return tf_dist_strategy, tf_dist_strategy.num_replicas_in_sync
