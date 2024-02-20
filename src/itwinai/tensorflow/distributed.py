import tensorflow as tf

def get_strategy():
    """Strategy for distributed TensorFlow training"""
    implementation = tf.distribute.experimental.CommunicationImplementation.NCCL
    communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation)

    # declare distribution strategy
    tf_dist_strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)

    # get total number of workers
    print("Number of devices: {}".format(tf_dist_strategy.num_replicas_in_sync))

    return tf_dist_strategy, tf_dist_strategy.num_replicas_in_sync
