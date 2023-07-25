from os.path import join, exists
from os import listdir, makedirs
from datetime import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import logging
import gdown

from lib.macros import PATCH_SIZE as patch_size, SHAPE as shape
from lib.models.setup import get_network_config, load_model
from lib.transform import coo_left_right, coo_up_down, coo_rot180, msk_left_right, msk_up_down, msk_rot180
from lib.macros import PatchType, Network, Losses, RegularizationStrength, Activation, LabelNoCyclone, AugmentationType
from lib.tfrecords.functions import read_tfrecord_as_tensor
from lib.macros import DRV_VARS_1, COO_VARS_1, MSK_VAR_1
from lib.tfrecords.dataset import eFlowsTFRecordDataset
from lib.strategy import get_mirrored_strategy
from lib.callbacks import ProcessBenchmark
from lib.scaling import save_tf_minmax
from lib.utils import Timer, saveparams
from trainer import CyclonesTrainer


def trainval(args):
    # get starting time of the script so that we can easily find run parameters and results.
    FORMATTED_DATETIME = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    MAIN_DIR = args.main_dir
    DATA_DIR = join(MAIN_DIR, 'data/')
    MODEL_BACKUP_DIR = join(MAIN_DIR, 'models/')
    DATASET_DIR = join(DATA_DIR, 'tfrecords', 'trainval/')
    LIB_DIR = join(MAIN_DIR, 'lib/')
    EXPERIMENTS_DIR = join(MAIN_DIR, 'experiments')
    RUN_DIR = join(EXPERIMENTS_DIR, args.run_name +'_'+FORMATTED_DATETIME)

    # define timer names
    tot_exec_timer = 'tot_exec_elapsed_time'
    io_timer = 'io_elapsed_time'
    train_timer = 'training_elapsed_time'

    # running time setup
    get_time = Timer(timers=[tot_exec_timer, io_timer, train_timer])
    get_time.start(tot_exec_timer)

    # Trainer from CLI args
    trainer = CyclonesTrainer(
        MAIN_DIR,
        args.epochs,
        args.network,
        args.activation,
        args.regularization_strength,
        args.learning_rate,
        args.loss,
        [len(DRV_VARS_1), len(COO_VARS_1)],
        args.batch_size,
        patch_size,
        args.kernel_size,
    )

    # CLI argument parsing
    regularization_strength, regularizer = [rg.value for rg in RegularizationStrength if rg.name.lower() == args.regularization_strength][0]
    experiment, (drv_vars, coo_vars, msk_var, channels) = 'default', (DRV_VARS_1, COO_VARS_1, MSK_VAR_1, [len(DRV_VARS_1), len(COO_VARS_1)])
    loss_name, loss = [l.value for l in Losses if l.name.lower() == args.loss][0]
    label_no_cyclone = args.label_no_cyclone
    shuffle_buffer = args.shuffle_buffer
    learning_rate = args.learning_rate
    target_scale = args.target_scale
    model_backup = args.model_backup
    kernel_size = args.kernel_size
    batch_size = args.batch_size
    patch_type = args.patch_type
    activation = args.activation
    augment = args.augmentation
    run_name = args.run_name
    aug_type = args.aug_type
    network = args.network
    shuffle = args.shuffle
    epochs = args.epochs
    cores = args.cores

    # additional hyperparameters
    split_ratio = (0.75, 0.25) 
    feature_range = (0,1)
    save_freq = 20
    
    #################################################################################################
    #                                                                                               #
    #                                       SOURCES AND PATHS                                       #
    #                                                                                               #
    #################################################################################################

    # directories setup
    SCALER_DIR = join(RUN_DIR, 'scalers')
    TENSORBOARD_DIR = join(RUN_DIR,'tensorboard')
    CHECKPOINTS_DIR = join(RUN_DIR,'checkpoints')

    makedirs(EXPERIMENTS_DIR, exist_ok=True)
    makedirs(RUN_DIR, exist_ok=True)
    makedirs(SCALER_DIR, exist_ok=True)
    makedirs(TENSORBOARD_DIR, exist_ok=True)
    makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # files and csvs definition
    LOG_FILE = join(RUN_DIR,'run.log')
    CHECKPOINTS_FILEPATH = join(CHECKPOINTS_DIR, 'model_{epoch:02d}.h5')
    PATH_HYPERPARAMETERS_CSV = join(EXPERIMENTS_DIR, 'models_hyperparameters.csv')
    LOSS_METRICS_HISTORY_CSV = join(RUN_DIR, 'loss_metrics_history.csv')
    BENCHMARK_HISTORY_CSV = join(RUN_DIR, 'benchmark_history.csv')
    TRAINVAL_TIME_CSV = join(RUN_DIR, 'trainval_time.csv')
    HYPERPARAMETERS_DUMP = join(RUN_DIR, 'hyperpars.dump')
    SCALER_FILE = join(SCALER_DIR, 'minmax.tfrecord')

    # initialize logger
    logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", level=logging.DEBUG, filename=LOG_FILE, datefmt='%Y-%m-%d %H:%M:%S')

    # type of patches
    if augment:
        if msk_var:
            aug_fns = {'left_right':msk_left_right,'up_down':msk_up_down,'rot180':msk_rot180}
        else:
            aug_fns = {'left_right':coo_left_right,'up_down':coo_up_down,'rot180':coo_rot180}
    else: 
        aug_fns = {}

    # log
    logging.debug(f'Passed CLI Arguments : \n\tregularization_strength={regularization_strength} \n\tlabel_no_cyclone={label_no_cyclone} \n\tshuffle_buffer={shuffle_buffer} \n\tlearning_rate={learning_rate} \n\tbatch_size={batch_size} \n\tpatch_type={patch_type} \n\texperiment={experiment} \n\tactivation={activation} \n\taugment={augment}={aug_fns} \n\trun_name={run_name} \n\tnetwork={network} \n\tshuffle={shuffle} \n\tepochs={epochs} \n\tcores={cores} \n\tloss={loss}')

    #################################################################################################
    #                                                                                               #
    #                                    MODEL INFRASTRUCTURE                                       #
    #                                                                                               #
    #################################################################################################

    # model infrastructure
    get_time.start(io_timer)

    columns_model_hyperparameters_df = ['date_time', 'epochs', 'batch_size', 'shuffle', 'norm_output', 'norm_min', 'norm_max', 'train_percent', 'val_percent', 'opt', 'loss', 'weight_initializer', 'activation', 'aug_dict', 'l1', 'l2']
    if not exists(PATH_HYPERPARAMETERS_CSV):
        pd.DataFrame(columns=columns_model_hyperparameters_df).to_csv(PATH_HYPERPARAMETERS_CSV)
    df_hypp = pd.read_csv(PATH_HYPERPARAMETERS_CSV, dtype='str', usecols=columns_model_hyperparameters_df)

    get_time.stop(io_timer)

    #################################################################################################
    #                                                                                               #
    #                                   TRAINING HYPERPARAMETERS                                    #
    #                                                                                               #
    #################################################################################################
    # log
    logging.debug(f'Program started')

    get_time.start(io_timer)

    # save hyperparameters
    saveparams(
        HYPERPARAMETERS_DUMP, regularization_strength=regularization_strength, label_no_cyclone=label_no_cyclone,
        shuffle_buffer=shuffle_buffer, learning_rate=learning_rate, batch_size=batch_size, patch_type=patch_type,
        experiment=experiment, activation=activation, augment=augment, run_name=run_name, network=network, shuffle=shuffle,
        epochs=epochs, cores=cores, loss_name=loss_name, loss=loss, split_ratio=split_ratio, feature_range=feature_range,
        save_freq=save_freq, drv_vars=drv_vars, coo_vars=coo_vars, msk_var=msk_var, channels=channels, aug_type=aug_type,
        patch_size=patch_size, target_scale=target_scale, kernel_size=kernel_size
    )

    # download dataset
    url = "https://drive.google.com/drive/folders/15DEq33MmtRvIpe2bNCg44lnfvEiHcPaf"
    gdown.download_folder(url=url, quiet=False, output=join(MAIN_DIR, 'data'))

    # get records filenames
    cyclone_files = sorted([join(DATASET_DIR,f) for f in listdir(DATASET_DIR) if f.endswith('.tfrecord') and f.startswith(PatchType.CYCLONE.value)])
    if patch_type == PatchType.NEAREST.value:
        adj_files = sorted([join(DATASET_DIR,f) for f in listdir(DATASET_DIR) if f.endswith('.tfrecord') and f.startswith(PatchType.NEAREST.value)])
    elif patch_type == PatchType.ALLADJACENT.value:
        adj_files = sorted([join(DATASET_DIR,f) for f in listdir(DATASET_DIR) if f.endswith('.tfrecord') and f.startswith(PatchType.ALLADJACENT.value)])
    random_files = sorted([join(DATASET_DIR,f) for f in listdir(DATASET_DIR) if f.endswith('.tfrecord') and f.startswith(PatchType.RANDOM.value)])

    get_time.stop(io_timer)

    # shuffle the data
    if shuffle:
        np.random.shuffle(cyclone_files)
        np.random.shuffle(adj_files)
        np.random.shuffle(random_files)

    def split_files(files, ratio):
        n = len(files)
        return (files[0:int(ratio[0]*n)], files[int(ratio[0]*n):int((ratio[0]+ratio[1])*n)])

    # divide into train, valid and test dataset files
    train_c_fs, valid_c_fs = split_files(files=cyclone_files, ratio=split_ratio)
    train_a_fs, valid_a_fs = split_files(files=adj_files, ratio=split_ratio)
    train_r_fs, valid_r_fs = split_files(files=random_files, ratio=split_ratio)

    # merge all the files together
    train_files = train_c_fs + train_a_fs + train_r_fs
    valid_files = valid_c_fs + valid_a_fs + valid_r_fs

    # shuffle the data
    #if shuffle:
    #    np.random.shuffle(train_files)
    #    np.random.shuffle(valid_files)

    # log
    logging.debug(f'Train, valid and test data files loaded. We have {len(train_files)} - {len(valid_files)} shard-files for training and validation.')

    get_time.start(io_timer)

    # compute scaler on training data
    Xt, _ = read_tfrecord_as_tensor(filenames=train_files, shape=shape, drv_vars=drv_vars, coo_vars=coo_vars, msk_var=msk_var)
    X_scaler = save_tf_minmax(Xt.numpy(), outfile=SCALER_FILE)
    scalers = [X_scaler, None]
    Xt = None

    get_time.stop(io_timer)

    # instantiate training, validation and test sets
    # Form: (dataset, dataset_size) = eFlowsTFRecordDataset...
    train = eFlowsTFRecordDataset(cyc_fnames=train_c_fs, adj_fnames=train_a_fs, rnd_fnames=train_r_fs, batch_size=batch_size, epochs=epochs, scalers=scalers, target_scale=target_scale, shape=shape, label_no_cyclone=label_no_cyclone, drv_vars=drv_vars, coo_vars=coo_vars, msk_var=msk_var, shuffle_buffer=shuffle_buffer, aug_fns=aug_fns, patch_type=patch_type, aug_type=aug_type)
    test = eFlowsTFRecordDataset(cyc_fnames=valid_c_fs, adj_fnames=valid_a_fs, rnd_fnames=valid_r_fs, batch_size=batch_size, epochs=epochs, scalers=scalers, target_scale=target_scale, shape=shape, label_no_cyclone=label_no_cyclone, drv_vars=drv_vars, coo_vars=coo_vars, msk_var=msk_var, shuffle_buffer=shuffle_buffer, aug_fns=aug_fns, patch_type=patch_type, aug_type=aug_type)

    # Train
    trainer.train([train, test])

    # save run hyperparameters
    # df_hypp.loc[len(df_hypp.index)] = [FORMATTED_DATETIME, epochs, batch_size, shuffle, int(False), feature_range[0], feature_range[1], split_ratio[0], split_ratio[1], optimizer.get_config(), loss, 0, activation, aug_fns, 0, 0]
    # df_hypp.to_csv(PATH_HYPERPARAMETERS_CSV)

    # log
    logging.debug(f'Saved run hyperparameters history')

    # save trainval execution times
    pd.DataFrame(data={
        "Training exec time" : [get_time.exec_times[train_timer]],
        "I/O exec time" : [get_time.exec_times[io_timer]],
        "Total Execution Time" : [get_time.exec_times[tot_exec_timer]]
    }).to_csv(TRAINVAL_TIME_CSV)

    # log
    logging.debug(f'Process completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='trainval.py', description="Train a VGG custom model on eFlows Dataset")
    
    # REQUIRED ARGUMENTS
    parser.add_argument( "-bs", "--batch_size", type=int, help="Global batch size of data", required=True)
    parser.add_argument( "-e", "--epochs", type=int, help="Number of epochs through which the model must be trained", required=True)
    parser.add_argument( "-md", "--main_dir", type=str, help="Main directory for the project", required=True)

    # OPTIONAL ARGUMENTS
    parser.add_argument( "-rn", "--run_name", default='debug', help="Name to be assigned to the run", required=False)
    parser.add_argument( "-mb", "--model_backup", default=None, help="The filepath to a trained model to be loaded", required=False)
    parser.add_argument( "-ks", "--kernel_size", default=None, type=int, help="Kernel size (only for Model V5)", required=False)
    parser.add_argument( "-s", "--shuffle", default='False', help="Number of consecutive samples to be shuffled", required=False)
    parser.add_argument( "-a", "--augmentation", default='True', help="Whether or not to perform data augmentation", required=False)
    parser.add_argument( "-c", "--cores", default=None, type=int, help="Number of cores (for local mirrored strategy)", required=False)
    parser.add_argument( "-sb", "--shuffle_buffer", default=None, type=int, help="Number of consecutive samples to be shuffled", required=False)
    parser.add_argument( "-lr", "--learning_rate", default=0.0001, type=float, help="Learning rate at which the model is trained", required=False)
    parser.add_argument( "-ts", "--target_scale", default='False', choices=['True','False'], help="Whether or not to scale the target", required=False)
    parser.add_argument( "-l", "--loss", default=Losses.MAE.value[0], choices=[l.value[0] for l in Losses], help="Loss function to be applied", required=False)
    parser.add_argument( "-n", "--network", default=Network.VGG_V1.value, choices=[n.value for n in Network], help="Neural network used to train the model", required=False)
    parser.add_argument( "-ac", "--activation", default=Activation.LINEAR.value, choices=[a.value for a in Activation], help="Last layer activation function", required=False)
    parser.add_argument( "-at", "--aug_type", default=AugmentationType.ONLY_TCS.value, choices=[at.value for at in AugmentationType], help="Type of augmentation", required=False)
    parser.add_argument( "-pt", "--patch_type", default=PatchType.NEAREST.value, choices=[pt.value for pt in PatchType], help="Type of patches used during training", required=False)
    parser.add_argument( "-lc", "--label_no_cyclone", default=str(LabelNoCyclone.NONE.value), choices=[str(lnc.value) for lnc in LabelNoCyclone], help="The label assigned to the cyclone", required=False)
    parser.add_argument( "-rg", "--regularization_strength", default=RegularizationStrength.NONE.value[0], choices=[r.value[0] for r in RegularizationStrength], help="Regularization strength", required=False)

    args = parser.parse_args()

    if args.shuffle == 'True':
        args.shuffle = True
    elif args.shuffle == 'False':
        args.shuffle = False
    
    if args.target_scale == 'True':
        args.target_scale = True
    elif args.target_scale == 'False':
        args.target_scale = False

    if args.augmentation == 'True':
        args.augmentation = True
    elif args.augmentation == 'False':
        args.augmentation = False
    
    if args.label_no_cyclone == 'None':
        args.label_no_cyclone = LabelNoCyclone.NONE.value
    else:
        args.label_no_cyclone = float(args.label_no_cyclone)
    
    trainval(args)

