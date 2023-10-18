import logging
from os import listdir
from os.path import join, exists
from itwinai.components import DataGetter
from typing import List, Dict, Optional, Tuple
from lib.macros import (
    PatchType,
    LabelNoCyclone,
    AugmentationType,
)

import gdown
import numpy as np


from lib.tfrecords.functions import read_tfrecord_as_tensor
from lib.scaling import save_tf_minmax
from lib.tfrecords.dataset import eFlowsTFRecordDataset
from lib.transform import (
    coo_left_right,
    coo_up_down,
    coo_rot180,
    msk_left_right,
    msk_up_down,
    msk_rot180,
)


class TensorflowDataGetter(DataGetter):
    def __init__(
        self,
        patch_type: PatchType,
        shuffle: bool,
        split_ratio: List[float],
        batch_size: int,
        augment: bool,
        epochs: int,
        target_scale: bool,
        label_no_cyclone: LabelNoCyclone,
        aug_type: AugmentationType,
        experiment: dict,
        shuffle_buffer: int = None,
        data_path: str = "tmp_data"
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.epochs = epochs
        self.target_scale = target_scale
        self.label_no_cyclone = label_no_cyclone.value
        self.shuffle_buffer = shuffle_buffer
        self.aug_type = aug_type.value
        self.patch_type = patch_type.value
        self.augment = augment
        self.shuffle = shuffle
        self.data_path = data_path
        self.drv_vars, self.coo_vars = (
            experiment["DRV_VARS_1"],
            experiment["COO_VARS_1"],
        )
        self.msk_var = (
            None if experiment["MSK_VAR_1"] == "None"
            else experiment["MSK_VAR_!"]
        )
        self.channels = [len(self.drv_vars), len(self.coo_vars)]

        # Shuffle
        if shuffle:
            np.random.shuffle(self.cyclone_files)
            np.random.shuffle(self.adj_files)
            np.random.shuffle(self.random_files)

        # Patches types
        if self.augment:
            if self.msk_var:
                self.aug_fns = {
                    "left_right": msk_left_right,
                    "up_down": msk_up_down,
                    "rot180": msk_rot180,
                }
            else:
                self.aug_fns = {
                    "left_right": coo_left_right,
                    "up_down": coo_up_down,
                    "rot180": coo_rot180,
                }
        else:
            self.aug_fns = {}

    def split_files(self, files, ratio):
        n = len(files)
        return (
            files[0: int(ratio[0] * n)],
            files[int(ratio[0] * n): int((ratio[0] + ratio[1]) * n)],
        )

    def load(self):
        # divide into train, valid and test dataset files
        train_c_fs, valid_c_fs = self.split_files(
            files=self.cyclone_files, ratio=self.split_ratio
        )
        train_a_fs, valid_a_fs = self.split_files(
            files=self.adj_files, ratio=self.split_ratio
        )
        train_r_fs, valid_r_fs = self.split_files(
            files=self.random_files, ratio=self.split_ratio
        )

        # merge all the files together
        train_files = train_c_fs + train_a_fs + train_r_fs
        # valid_files = valid_c_fs + valid_a_fs + valid_r_fs

        # compute scaler on training data
        Xt, _ = read_tfrecord_as_tensor(
            filenames=train_files,
            shape=self.shape,
            drv_vars=self.drv_vars,
            coo_vars=self.coo_vars,
            msk_var=self.msk_var,
        )
        X_scaler = save_tf_minmax(Xt.numpy(), outfile=self.scaler_file)
        scalers = [X_scaler, None]
        Xt = None

        # instantiate training, validation and test sets
        # Contains: (dataset, n_count)
        train_dataset = eFlowsTFRecordDataset(
            cyc_fnames=train_c_fs,
            adj_fnames=train_a_fs,
            rnd_fnames=train_r_fs,
            batch_size=self.batch_size,
            epochs=self.epochs,
            scalers=scalers,
            target_scale=self.target_scale,
            shape=self.shape,
            label_no_cyclone=self.label_no_cyclone,
            drv_vars=self.drv_vars,
            coo_vars=self.coo_vars,
            msk_var=self.msk_var,
            shuffle_buffer=self.shuffle_buffer,
            aug_fns=self.aug_fns,
            patch_type=self.patch_type,
            aug_type=self.aug_type,
        )
        valid_dataset = eFlowsTFRecordDataset(
            cyc_fnames=valid_c_fs,
            adj_fnames=valid_a_fs,
            rnd_fnames=valid_r_fs,
            batch_size=self.batch_size,
            epochs=self.epochs,
            scalers=scalers,
            target_scale=self.target_scale,
            shape=self.shape,
            label_no_cyclone=self.label_no_cyclone,
            drv_vars=self.drv_vars,
            coo_vars=self.coo_vars,
            msk_var=self.msk_var,
            shuffle_buffer=self.shuffle_buffer,
            aug_fns=self.aug_fns,
            patch_type=self.patch_type,
            aug_type=self.aug_type,
        )
        return train_dataset, valid_dataset

    def execute(
        self,
        config: Optional[Dict] = None
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        config = self.setup_config(config)
        train, test = self.load()
        logging.debug("Train, valid and test datasets loaded.")
        return (train, test), config

    def setup_config(self, config: Optional[Dict] = None) -> Dict:
        config = config if config is not None else {}
        self.shape = config["shape"]
        root_dir = config["root_dir"]

        # Download data
        url = (
            "https://drive.google.com/drive/folders/"
            "15DEq33MmtRvIpe2bNCg44lnfvEiHcPaf"
        )
        if not exists(join(root_dir, self.data_path)):
            gdown.download_folder(
                url=url, quiet=False,
                output=join(root_dir, self.data_path)
            )

        # Scalar fields
        self.root_dir = root_dir
        self.dataset_dir = join(root_dir, self.data_path,
                                "tfrecords", "trainval/")
        self.scaler_file = join(config["scaler_dir"], "minmax.tfrecord")

        # get records filenames
        self.cyclone_files = sorted(
            [
                join(self.dataset_dir, f)
                for f in listdir(self.dataset_dir)
                if f.endswith(".tfrecord") and f.startswith(
                    PatchType.CYCLONE.value)
            ]
        )
        if self.patch_type == PatchType.NEAREST.value:
            self.adj_files = sorted(
                [
                    join(self.dataset_dir, f)
                    for f in listdir(self.dataset_dir)
                    if f.endswith(".tfrecord") and f.startswith(
                        PatchType.NEAREST.value)
                ]
            )
        elif self.patch_type == PatchType.ALLADJACENT.value:
            self.adj_files = sorted(
                [
                    join(self.dataset_dir, f)
                    for f in listdir(self.dataset_dir)
                    if f.endswith(".tfrecord")
                    and f.startswith(PatchType.ALLADJACENT.value)
                ]
            )
        self.random_files = sorted(
            [
                join(self.dataset_dir, f)
                for f in listdir(self.dataset_dir)
                if f.endswith(".tfrecord") and f.startswith(
                    PatchType.RANDOM.value)
            ]
        )

        config["epochs"] = self.epochs
        config["batch_size"] = self.batch_size
        config["channels"] = self.channels
        return config
