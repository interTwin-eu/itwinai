import tensorflow as tf

def get_strategy():
    """Strategy for distributed TensorFlow training"""
    cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=12345)
    implementation = tf.distribute.experimental.CommunicationImplementation.NCCL
    communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation)

    # declare distribution strategy
    tf_dist_strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver = cluster_resolver, communication_options=communication_options)

    # task id from cluster resolver
    task_info = cluster_resolver.get_task_info()
    task_id = task_info[1]
    
    # number of workers
    n_workers = int(os.environ['SLURM_NTASKS'])
    # list of devices per worker
    devices = tf.config.experimental.list_physical_devices('GPU')
    # number of devices per worker
    n_gpus_per_worker = len(devices)
    # total number of GPUs
    n_gpus = n_workers * n_gpus_per_worker

    # get total number of workers
    print("Number of devices: {}".format(tf_dist_strategy.num_replicas_in_sync))

    return tf_dist_strategy, tf_dist_strategy.num_replicas_in_sync
