import tensorflow as tf


# gets the mirrored strategy based on whether or not we are running the model
# with CPU or GPU
def get_mirrored_strategy(cores=4):
    if cores:
        CPUs = ['CPU:'+str(i) for i in range(cores)]
        mirrored_strategy = tf.distribute.MirroredStrategy(CPUs)
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()

    print('Number of devices: {}'.format(
        mirrored_strategy.num_replicas_in_sync))

    return mirrored_strategy, mirrored_strategy.num_replicas_in_sync
