import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import gdown

from os.path import join, exists
from os import listdir, makedirs
from datetime import datetime

from lib.macros import PATCH_SIZE as patch_size, SHAPE as shape
from lib.transform import coo_left_right, coo_up_down, coo_rot180, msk_left_right, msk_up_down, msk_rot180
from lib.macros import PatchType, Network, Losses, RegularizationStrength, Activation, LabelNoCyclone, AugmentationType
from lib.tfrecords.functions import read_tfrecord_as_tensor
from lib.macros import DRV_VARS_1, COO_VARS_1, MSK_VAR_1
from lib.tfrecords.dataset import eFlowsTFRecordDataset
from lib.strategy import get_mirrored_strategy
from lib.callbacks import ProcessBenchmark
from lib.scaling import save_tf_minmax
from lib.utils import Timer, saveparams, get_network_config, load_model
from itwinai.backend.components import Trainer

class TensorflowTrainer(Trainer):
    def __init__(
            self,
            batch_size,
            epochs,
            run_name,
            model_backup,
            kernel_size,
            shuffle,
            augmentation,
            cores,
            shuffle_buffer,
            learning_rate,
            target_scale,
            loss,
            network,
            activation,
            aug_type,
            patch_type,
            label_no_cyclone,
            regularization_strength,
            MAIN_DIR
    ):
        # get starting time of the script so that we can easily find run parameters and results.
        self.FORMATTED_DATETIME = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.MAIN_DIR = MAIN_DIR
        self.DATA_DIR = join(MAIN_DIR, 'data/')
        self.MODEL_BACKUP_DIR = join(MAIN_DIR, 'models/')
        self.DATASET_DIR = join(self.DATA_DIR, 'tfrecords', 'trainval/')

        self.batch_size = batch_size
        self.epochs = epochs
        self.run_name = run_name
        self.model_backup = model_backup
        self.kernel_size = kernel_size
        self.shuffle = shuffle
        self.augment = augmentation
        self.cores = cores
        self.shuffle_buffer = shuffle_buffer
        self.learning_rate = learning_rate
        self.target_scale = target_scale
        self.loss = loss
        self.network = network
        self.activation = activation
        self.aug_type = aug_type
        self.patch_type = patch_type
        self.label_no_cyclone = label_no_cyclone
        # CLI argument parsing
        self.regularization_strength, self.regularizer = \
        [rg.value for rg in RegularizationStrength if rg.name.lower() == regularization_strength][0]
        self.experiment, (self.drv_vars, self.coo_vars, self.msk_var, self.channels) = 'default', (DRV_VARS_1, COO_VARS_1, MSK_VAR_1, [len(DRV_VARS_1), len(COO_VARS_1)])
        self.loss_name, self.loss = [l.value for l in Losses if l.name.lower() == loss][0]

        if args.model_backup is not None and not exists(join(self.MODEL_BACKUP_DIR, args.model_backup)):
            raise FileNotFoundError('Model backup file not found')

    def execute(self, args):
        pass

    def setup(self, args):
        pass

    def train(self):
        # define timer names
        tot_exec_timer = 'tot_exec_elapsed_time'
        io_timer = 'io_elapsed_time'
        train_timer = 'training_elapsed_time'

        # running time setup
        get_time = Timer(timers=[tot_exec_timer, io_timer, train_timer])
        get_time.start(tot_exec_timer)

        # additional hyperparameters
        split_ratio = (0.75, 0.25)
        feature_range = (0, 1)
        save_freq = 20

        #################################################################################################
        #                                                                                               #
        #                                       SOURCES AND PATHS                                       #
        #                                                                                               #
        #################################################################################################

        # directories setup
        EXPERIMENTS_DIR = join(self.MAIN_DIR, 'experiments')
        RUN_DIR = join(EXPERIMENTS_DIR, self.run_name + '_' + self.FORMATTED_DATETIME)
        SCALER_DIR = join(RUN_DIR, 'scalers')
        TENSORBOARD_DIR = join(RUN_DIR, 'tensorboard')
        CHECKPOINTS_DIR = join(RUN_DIR, 'checkpoints')

        makedirs(EXPERIMENTS_DIR, exist_ok=True)
        makedirs(RUN_DIR, exist_ok=True)
        makedirs(SCALER_DIR, exist_ok=True)
        makedirs(TENSORBOARD_DIR, exist_ok=True)
        makedirs(CHECKPOINTS_DIR, exist_ok=True)

        # files and csvs definition
        LOG_FILE = join(RUN_DIR, 'run.log')
        CHECKPOINTS_FILEPATH = join(CHECKPOINTS_DIR, 'model_{epoch:02d}.h5')
        PATH_HYPERPARAMETERS_CSV = join(EXPERIMENTS_DIR, 'models_hyperparameters.csv')
        LOSS_METRICS_HISTORY_CSV = join(RUN_DIR, 'loss_metrics_history.csv')
        BENCHMARK_HISTORY_CSV = join(RUN_DIR, 'benchmark_history.csv')
        TRAINVAL_TIME_CSV = join(RUN_DIR, 'trainval_time.csv')
        HYPERPARAMETERS_DUMP = join(RUN_DIR, 'hyperpars.dump')
        SCALER_FILE = join(SCALER_DIR, 'minmax.tfrecord')

        # download files
        url = "https://drive.google.com/drive/folders/15DEq33MmtRvIpe2bNCg44lnfvEiHcPaf"
        gdown.download_folder(url=url, quiet=False, output=self.MAIN_DIR)

        # initialize logger
        logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", level=logging.DEBUG, filename=LOG_FILE,
                            datefmt='%Y-%m-%d %H:%M:%S')

        # type of patches
        if self.augment:
            if self.msk_var:
                self.aug_fns = {'left_right': msk_left_right, 'up_down': msk_up_down, 'rot180': msk_rot180}
            else:
                self.aug_fns = {'left_right': coo_left_right, 'up_down': coo_up_down, 'rot180': coo_rot180}
        else:
            self.aug_fns = {}

        # log
        logging.debug(
            f'Passed CLI Arguments : \n\tregularization_strength={self.regularization_strength} '
            f'\n\tlabel_no_cyclone={self.label_no_cyclone} '
            f'\n\tshuffle_buffer={self.shuffle_buffer} '
            f'\n\tlearning_rate={self.learning_rate} '
            f'\n\tbatch_size={self.batch_size} '
            f'\n\tpatch_type={self.patch_type} '
            f'\n\texperiment={self.experiment} '
            f'\n\tactivation={self.activation} '
            f'\n\taugment={self.augment}={self.aug_fns} '
            f'\n\trun_name={self.run_name} '
            f'\n\tnetwork={self.network} '
            f'\n\tshuffle={self.shuffle} '
            f'\n\tepochs={self.epochs} '
            f'\n\tcores={self.cores} '
            f'\n\tloss={self.loss}'
        )

        #################################################################################################
        #                                                                                               #
        #                                    MODEL INFRASTRUCTURE                                       #
        #                                                                                               #
        #################################################################################################

        # model infrastructure
        get_time.start(io_timer)

        columns_model_hyperparameters_df = ['date_time', 'epochs', 'batch_size', 'shuffle', 'norm_output', 'norm_min',
                                            'norm_max', 'train_percent', 'val_percent', 'opt', 'loss',
                                            'weight_initializer', 'activation', 'aug_dict', 'l1', 'l2']
        if not exists(PATH_HYPERPARAMETERS_CSV):
            pd.DataFrame(columns=columns_model_hyperparameters_df).to_csv(PATH_HYPERPARAMETERS_CSV)
        df_hypp = pd.read_csv(PATH_HYPERPARAMETERS_CSV, dtype='str', usecols=columns_model_hyperparameters_df)

        get_time.stop(io_timer)

        #################################################################################################
        #                                                                                               #
        #                                   TRAINING HYPERPARAMETERS                                    #
        #                                                                                               #
        #################################################################################################

        # hyperparameters
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)  # 0.001)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, min_delta=0.0001,
                                             restore_best_weights=True, verbose=1, mode='min'),
            tf.keras.callbacks.CSVLogger(LOSS_METRICS_HISTORY_CSV),
            # tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, histogram_freq=1, write_images=True, write_graph=True, update_freq='batch', profile_batch='1,500'),
            ProcessBenchmark(BENCHMARK_HISTORY_CSV),
        ]
        last_model_name = join(RUN_DIR, 'last_model.h5')
        if self.model_backup:
            best_model_name = join(self.MODEL_BACKUP_DIR, self.model_backup, 'best_model.h5')

        # log
        logging.debug(f'Program started')

        get_time.start(io_timer)

        # save hyperparameters
        saveparams(
            HYPERPARAMETERS_DUMP, regularization_strength=self.regularization_strength, label_no_cyclone=self.label_no_cyclone,
            shuffle_buffer=self.shuffle_buffer, learning_rate=self.learning_rate, batch_size=self.batch_size, patch_type=self.patch_type,
            experiment=self.experiment, activation=self.activation, augment=self.augment, run_name=self.run_name, network=self.network,
            shuffle=self.shuffle,
            epochs=self.epochs, cores=self.cores, loss_name=self.loss_name, loss=self.loss, split_ratio=split_ratio,
            feature_range=feature_range,
            save_freq=save_freq, drv_vars=self.drv_vars, coo_vars=self.coo_vars, msk_var=self.msk_var, channels=self.channels,
            aug_type=self.aug_type,
            patch_size=patch_size, target_scale=self.target_scale, kernel_size=self.kernel_size
        )

        # get records filenames
        cyclone_files = sorted([join(self.DATASET_DIR, f) for f in listdir(self.DATASET_DIR) if
                                f.endswith('.tfrecord') and f.startswith(PatchType.CYCLONE.value)])
        if self.patch_type == PatchType.NEAREST.value:
            adj_files = sorted([join(self.DATASET_DIR, f) for f in listdir(self.DATASET_DIR) if
                                f.endswith('.tfrecord') and f.startswith(PatchType.NEAREST.value)])
        elif self.patch_type == PatchType.ALLADJACENT.value:
            adj_files = sorted([join(self.DATASET_DIR, f) for f in listdir(self.DATASET_DIR) if
                                f.endswith('.tfrecord') and f.startswith(PatchType.ALLADJACENT.value)])
        random_files = sorted([join(self.DATASET_DIR, f) for f in listdir(self.DATASET_DIR) if
                               f.endswith('.tfrecord') and f.startswith(PatchType.RANDOM.value)])

        get_time.stop(io_timer)

        # shuffle the data
        if self.shuffle:
            np.random.shuffle(cyclone_files)
            np.random.shuffle(adj_files)
            np.random.shuffle(random_files)

        def split_files(files, ratio):
            n = len(files)
            return (files[0:int(ratio[0] * n)], files[int(ratio[0] * n):int((ratio[0] + ratio[1]) * n)])

        # divide into train, valid and test dataset files
        train_c_fs, valid_c_fs = split_files(files=cyclone_files, ratio=split_ratio)
        train_a_fs, valid_a_fs = split_files(files=adj_files, ratio=split_ratio)
        train_r_fs, valid_r_fs = split_files(files=random_files, ratio=split_ratio)

        # merge all the files together
        train_files = train_c_fs + train_a_fs + train_r_fs
        valid_files = valid_c_fs + valid_a_fs + valid_r_fs

        # shuffle the data
        # if shuffle:
        #    np.random.shuffle(train_files)
        #    np.random.shuffle(valid_files)

        # log
        logging.debug(
            f'Train, valid and test data files loaded. We have {len(train_files)} - {len(valid_files)} shard-files for training and validation.')

        get_time.start(io_timer)

        # compute scaler on training data
        Xt, _ = read_tfrecord_as_tensor(filenames=train_files, shape=shape, drv_vars=self.drv_vars, coo_vars=self.coo_vars,
                                        msk_var=self.msk_var)
        X_scaler = save_tf_minmax(Xt.numpy(), outfile=SCALER_FILE)
        scalers = [X_scaler, None]
        Xt = None

        get_time.stop(io_timer)

        # instantiate training, validation and test sets
        train_dataset, n_train = eFlowsTFRecordDataset(cyc_fnames=train_c_fs, adj_fnames=train_a_fs,
                                                       rnd_fnames=train_r_fs, batch_size=self.batch_size, epochs=self.epochs,
                                                       scalers=scalers, target_scale=self.target_scale, shape=shape,
                                                       label_no_cyclone=self.label_no_cyclone, drv_vars=self.drv_vars,
                                                       coo_vars=self.coo_vars, msk_var=self.msk_var,
                                                       shuffle_buffer=self.shuffle_buffer, aug_fns=self.aug_fns,
                                                       patch_type=self.patch_type, aug_type=self.aug_type)
        valid_dataset, n_valid = eFlowsTFRecordDataset(cyc_fnames=valid_c_fs, adj_fnames=valid_a_fs,
                                                       rnd_fnames=valid_r_fs, batch_size=self.batch_size, epochs=self.epochs,
                                                       scalers=scalers, target_scale=self.target_scale, shape=shape,
                                                       label_no_cyclone=self.label_no_cyclone, drv_vars=self.drv_vars,
                                                       coo_vars=self.coo_vars, msk_var=self.msk_var,
                                                       shuffle_buffer=self.shuffle_buffer, aug_fns=self.aug_fns,
                                                       patch_type=self.patch_type, aug_type=self.aug_type)

        # compute the steps per epoch for train and valid
        steps_per_epoch = n_train // self.batch_size
        validation_steps = n_valid // self.batch_size

        # log
        logging.debug(f'Train, valid and test datasets loaded.')

        # append the Model Checkpoint Callback to our callbacks
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINTS_FILEPATH,
                                                                       save_best_only=True, monitor='val_loss',
                                                                       mode='min', save_weights_only=False, verbose=1)
        callbacks.append(model_checkpoint_callback)

        get_time.start(train_timer)

        # set mirrored strategy
        mirrored_strategy, n_devices = get_mirrored_strategy(cores=self.cores)

        # log
        logging.debug(f'Mirrored strategy created with {n_devices} devices')

        # distribute datasets among MirroredStrategy's replicas
        dist_train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
        dist_valid_dataset = mirrored_strategy.experimental_distribute_dataset(valid_dataset)

        # inside the strategy load the model, data generators and train
        with mirrored_strategy.scope():
            # create model
            if not self.model_backup:  # we are not using model backup
                # we pass all the possible arguments, then the function will use only the required
                model = get_network_config(network=self.network, patch_size=self.patch_size, channels=self.channels,
                                           activation=self.activation, regularizer=self.regularizer,
                                           label_no_cyclone=self.label_no_cyclone, kernel_size=self.kernel_size)
                # log
                logging.debug(f'New model created')
            else:
                model = load_model(model_fpath=best_model_name)
                # log
                logging.debug(f'Model loaded from backup at {best_model_name}')

            # IMPORTANTE : le metriche vanno istanziate all'interno dello stesso scope di distribuzione (MirroredStrategy) in cui viene chiamata la model.compile()
            metrics = [tf.keras.metrics.MeanAbsoluteError(name='mae')]
            model.compile(loss=self.loss, optimizer=optimizer, metrics=metrics)

        # log
        logging.debug(f'Model compiled')

        # print model summary to check if model's architecture is correct
        print(model.summary())

        # train the model
        model.fit(dist_train_dataset,
                  validation_data=dist_valid_dataset,
                  steps_per_epoch=steps_per_epoch,
                  validation_steps=validation_steps,
                  epochs=self.epochs,
                  callbacks=callbacks
                  )

        get_time.stop(train_timer)

        # log
        logging.debug(f'Model trained')

        get_time.start(io_timer)

        # save the best model
        model.save(last_model_name)

        # log
        logging.debug(f'Saved training history')

        # save run hyperparameters
        df_hypp.loc[len(df_hypp.index)] = [self.FORMATTED_DATETIME, self.epochs, self.batch_size, self.shuffle, int(False),
                                           feature_range[0], feature_range[1], split_ratio[0], split_ratio[1],
                                           optimizer.get_config(), self.loss, 0, self.activation, self.aug_fns, 0, 0]
        df_hypp.to_csv(PATH_HYPERPARAMETERS_CSV)

        # log
        logging.debug(f'Saved run hyperparameters history')

        get_time.stop(io_timer)
        get_time.stop(tot_exec_timer)

        # save trainval execution times
        pd.DataFrame(data={
            "Training exec time": [get_time.exec_times[train_timer]],
            "I/O exec time": [get_time.exec_times[io_timer]],
            "Total Execution Time": [get_time.exec_times[tot_exec_timer]]
        }).to_csv(TRAINVAL_TIME_CSV)

        # log
        logging.debug(f'Process completed')


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='trainval.py', description="Train a VGG custom model on eFlows Dataset")

    # REQUIRED ARGUMENTS
    parser.add_argument("-bs", "--batch_size", type=int, help="Global batch size of data", required=True)
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs through which the model must be trained",
                        required=True)
    parser.add_argument("-md", "--MAIN_DIR", default=None, required=True)

    # OPTIONAL ARGUMENTS
    parser.add_argument("-rn", "--run_name", default='debug', help="Name to be assigned to the run", required=False)
    parser.add_argument("-mb", "--model_backup", default=None, help="The filepath to a trained model to be loaded",
                        required=False)
    parser.add_argument("-ks", "--kernel_size", default=None, type=int, help="Kernel size (only for Model V5)",
                        required=False)
    parser.add_argument("-s", "--shuffle", default='False', help="Number of consecutive samples to be shuffled",
                        required=False)
    parser.add_argument("-a", "--augmentation", default='True', help="Whether or not to perform data augmentation",
                        required=False)
    parser.add_argument("-c", "--cores", default=None, type=int, help="Number of cores (for local mirrored strategy)",
                        required=False)
    parser.add_argument("-sb", "--shuffle_buffer", default=None, type=int,
                        help="Number of consecutive samples to be shuffled", required=False)
    parser.add_argument("-lr", "--learning_rate", default=0.0001, type=float,
                        help="Learning rate at which the model is trained", required=False)
    parser.add_argument("-ts", "--target_scale", default='False', choices=['True', 'False'],
                        help="Whether or not to scale the target", required=False)
    parser.add_argument("-l", "--loss", default=Losses.MAE.value[0], choices=[l.value[0] for l in Losses],
                        help="Loss function to be applied", required=False)
    parser.add_argument("-n", "--network", default=Network.VGG_V1.value, choices=[n.value for n in Network],
                        help="Neural network used to train the model", required=False)
    parser.add_argument("-ac", "--activation", default=Activation.LINEAR.value, choices=[a.value for a in Activation],
                        help="Last layer activation function", required=False)
    parser.add_argument("-at", "--aug_type", default=AugmentationType.ONLY_TCS.value,
                        choices=[at.value for at in AugmentationType], help="Type of augmentation", required=False)
    parser.add_argument("-pt", "--patch_type", default=PatchType.NEAREST.value, choices=[pt.value for pt in PatchType],
                        help="Type of patches used during training", required=False)
    parser.add_argument("-lc", "--label_no_cyclone", default=str(LabelNoCyclone.NONE.value),
                        choices=[str(lnc.value) for lnc in LabelNoCyclone], help="The label assigned to the cyclone",
                        required=False)
    parser.add_argument("-rg", "--regularization_strength", default=RegularizationStrength.NONE.value[0],
                        choices=[r.value[0] for r in RegularizationStrength], help="Regularization strength",
                        required=False)

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

    trainer = TensorflowTrainer(
        batch_size=args.batch_size,
        epochs=args.epochs,
        run_name=args.run_name,
        model_backup=args.model_backup,
        kernel_size=args.kernel_size,
        shuffle=args.shuffle,
        augmentation=args.augmentation,
        cores=args.cores,
        shuffle_buffer=args.shuffle_buffer,
        learning_rate=args.learning_rate,
        target_scale=args.target_scale,
        loss=args.loss,
        network=args.network,
        activation=args.activation,
        aug_type=args.aug_type,
        patch_type=args.patch_type,
        label_no_cyclone=args.label_no_cyclone,
        regularization_strength=args.regularization_strength,
        MAIN_DIR=args.MAIN_DIR
    )
    trainer.train()