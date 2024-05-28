from typing import Optional
import tensorflow as tf

from .functions import (
    get_tensor_decoding_fn,
    get_scaling_fn,
    get_masking_fn, get_scale_target_fn
)
from ..macros import PatchType, AugmentationType


def get_interleave(cyc_weights, nocyc_weights):
    """
    Returns the interleaved dataset indexes based on cyclone and
    nocyclone weights.
    """
    # define cyclone interleave
    cyc_interleave = [i for i, w in enumerate(cyc_weights) for _ in range(w)]
    # define nocyclone interleave
    nocyc_interleave = [i+len(cyc_interleave)
                        for i, w in enumerate(nocyc_weights) for _ in range(w)]

    # compute the number of blocks + the remainder of the interleaves
    blocks = len(nocyc_interleave) // len(cyc_interleave)
    remainder = len(nocyc_interleave) % len(cyc_interleave)

    interleave = []
    for i in cyc_interleave:
        interleave += [i] + nocyc_interleave[i*blocks:(i+1)*blocks]
    if remainder:
        interleave += nocyc_interleave[-remainder:]

    return tf.cast(interleave, dtype=tf.int64)


def eFlowsTFRecordDataset(
    cyc_fnames, adj_fnames, rnd_fnames,  epochs,  # batch_size,
    scalers, target_scale=False, drv_vars=[], coo_vars=None,
    msk_var=None, shape=(40, 40),
    label_no_cyclone: Optional[float] = -0.3,
    shuffle_buffer=None, patch_type=PatchType.NEAREST.value,
    aug_type=AugmentationType.ONLY_TCS.value, aug_fns={},
    # drop_remainder=True
):
    # set autotune parameter to automatically manage resourches
    AUTOTUNE = tf.data.AUTOTUNE

    # compute the weight associated to an adjacent dataset
    adj_w = 3 if patch_type == PatchType.NEAREST.value else 8

    # setup dynamical lambda functions to be applied to this dataset
    tensor_decoding_fn = get_tensor_decoding_fn(
        shape=shape, drv_vars=drv_vars, coo_vars=coo_vars, msk_var=msk_var)
    scaling_fn = get_scaling_fn(scalers=scalers)
    masking_fn = get_masking_fn(mask=label_no_cyclone)
    scale_target_fn = get_scale_target_fn(
        label_no_cyclone=label_no_cyclone, patch_size=shape[0])

    # multiplier for the augmentation
    mul = 1 if not aug_fns else (len(aug_fns.keys()) + 1)

    # compute the number of samples into the dataset
    cyc_n_elems = sum([int(fname.split('/')[-1].split('.tfrecord')
                      [0].split('_')[-1]) for fname in cyc_fnames])
    rnd_n_elems = sum([int(fname.split('/')[-1].split('.tfrecord')
                      [0].split('_')[-1]) for fname in rnd_fnames])
    adj_n_elems = sum([int(fname.split('/')[-1].split('.tfrecord')
                      [0].split('_')[-1]) for fname in adj_fnames])
    n_elems = mul * cyc_n_elems + rnd_n_elems + adj_n_elems

    # total number of samples that will be yielded by this dataset
    count = n_elems * epochs

    # create standard datasets for each patch category
    cyc_dataset = tf.data.TFRecordDataset(
        cyc_fnames, num_parallel_reads=AUTOTUNE
    ).map(
        tensor_decoding_fn, num_parallel_calls=AUTOTUNE)
    rnd_dataset = tf.data.TFRecordDataset(
        rnd_fnames, num_parallel_reads=AUTOTUNE).map(
        tensor_decoding_fn, num_parallel_calls=AUTOTUNE)
    adj_dataset = tf.data.TFRecordDataset(
        adj_fnames, num_parallel_reads=AUTOTUNE).map(
        tensor_decoding_fn, num_parallel_calls=AUTOTUNE)

    # add cyclone dataset to the cyclone datasets
    cyc_datasets = [cyc_dataset]
    # add the weight of each cyclone dataset
    cyc_weights = [1]

    # add cyclone dataset to the nocyclone datasets
    nocyc_datasets = [rnd_dataset, adj_dataset]
    # add the weight of each nocyclone dataset
    nocyc_weights = [1, adj_w]

    # Create augmented TC datasets and interleave them to the original
    if aug_fns:
        #  augmentation of all patches
        if aug_type == AugmentationType.ALL_PATCHES.value:
            # define augmented datasets for each augmentation function
            aug_cyc_datasets = [
                (tf.data.TFRecordDataset(
                    cyc_fnames, num_parallel_reads=AUTOTUNE).map(
                        tensor_decoding_fn, num_parallel_calls=AUTOTUNE).map(
                    lambda x, y: (aug_fn((x, y))), num_parallel_calls=AUTOTUNE)
                 )
                for aug_fn in aug_fns.values()]

            aug_rnd_datasets = [(tf.data.TFRecordDataset(
                rnd_fnames, num_parallel_reads=AUTOTUNE).map(
                    tensor_decoding_fn, num_parallel_calls=AUTOTUNE).map(
                lambda x, y: (aug_fn((x, y))), num_parallel_calls=AUTOTUNE))
                for aug_fn in aug_fns.values()]

            aug_adj_datasets = [(tf.data.TFRecordDataset(
                adj_fnames, num_parallel_reads=AUTOTUNE).map(
                    tensor_decoding_fn, num_parallel_calls=AUTOTUNE).map(
                lambda x, y: (aug_fn((x, y))), num_parallel_calls=AUTOTUNE))
                for aug_fn in aug_fns.values()]

        # augmentation of only TC patches
        elif aug_type == AugmentationType.ONLY_TCS.value:
            # define augmented datasets for each augmentation function
            aug_cyc_datasets = [(tf.data.TFRecordDataset(
                cyc_fnames, num_parallel_reads=AUTOTUNE).map(
                    tensor_decoding_fn,
                    num_parallel_calls=AUTOTUNE).map(
                lambda x, y: (aug_fn((x, y))),
                num_parallel_calls=AUTOTUNE))
                for aug_fn in aug_fns.values()]
            aug_rnd_datasets = []
            aug_adj_datasets = []
    else:
        aug_cyc_datasets = []
        aug_rnd_datasets = []
        aug_adj_datasets = []

    # add weights from the augmented datasets
    cyc_weights += [1 for _ in range(len(aug_cyc_datasets))]
    nocyc_weights += [1 for _ in range(len(aug_rnd_datasets))]
    nocyc_weights += [adj_w for _ in range(len(aug_adj_datasets))]

    # add to cyclone and nocyclone datasets the augmentations
    cyc_datasets += aug_cyc_datasets
    nocyc_datasets += aug_rnd_datasets + aug_adj_datasets

    # create a list of all datasets to interleave on
    datasets = cyc_datasets + nocyc_datasets

    # get the interleave of all datasets
    interleave = get_interleave(
        cyc_weights=cyc_weights, nocyc_weights=nocyc_weights)

    # compute the choice dataset with the interleave
    choice_dataset = tf.data.Dataset.from_tensor_slices(
        interleave).repeat(count=count)

    # statically interleave elements from all the datasets
    dataset = tf.data.experimental.choose_from_datasets(
        datasets=datasets, choice_dataset=choice_dataset)

    # shuffle if necessary
    if shuffle_buffer:
        dataset = dataset.shuffle(
            shuffle_buffer, reshuffle_each_iteration=True)

    # NOTE: when running distributed training, the dataset
    # should be batched knowing the number of parallel
    # workers taking part to the pool! Skipping it now...
    # Example:
    #  batch_size = num_workers * worker_batch_size
    # # separate in batches
    # if batch_size:
    #     dataset = dataset.batch(
    #         batch_size, drop_remainder=drop_remainder,
    #         num_parallel_calls=AUTOTUNE)
    # else:
    #     dataset = dataset.batch(
    #         n_elems, drop_remainder=drop_remainder,
    #         num_parallel_calls=AUTOTUNE)

    # apply mask on target if label_no_cyclone is provided
    if label_no_cyclone:
        dataset = dataset.map(lambda X, y: (
            masking_fn((X, y))), num_parallel_calls=AUTOTUNE)

    # scale the data
    if scalers:
        dataset = dataset.map(lambda X, y: (
            scaling_fn((X, y))), num_parallel_calls=AUTOTUNE)
    if target_scale:
        dataset = dataset.map(lambda X, y: (
            scale_target_fn((X, y))), num_parallel_calls=AUTOTUNE)

    # set number of epochs that can be repeated on this dataset
    dataset = dataset.repeat(count=epochs)

    # add parallelism option
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
    options.experimental_threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)

    # prefetch
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    # return the dataset
    return dataset, n_elems
