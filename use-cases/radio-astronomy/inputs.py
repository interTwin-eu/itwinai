### MAIN SETTINGS ###

generate_data = True

### DATA GENERATION SETTINGS ###

tag = "train_v0_"
num_payloads = 50
plot_a_example = False
num_cpus = 1
param_folder = "syn_param/"
payload_folder = "syn_payload/"
image_shape = (128, 128)

### PATH SETTINGS ###

image_directory_npy = "./test_data/joint_dataset_8_classes_real_remix_128x128.npy"
label_directory_npy = "./test_data/joint_dataset_8_classes_real_remix_labels.npy"
image_tag = "train_v0_*_payload_detected.json"  #: '*' in the name is the index place holder of a image in the image set
image_directory = payload_folder
mask_tag = "train_v0_*_payload_flux.json"
mask_directory = payload_folder

unet_dir = "./models/trained_UNet_test_v0.pt"
fcnn_dir = "./models/trained_Filter_test_v0.pt"
cnn1d_dir = "./models/trained_CNN1D_test_v0.pt"
offset = 0
size_of_set = 10

### SPLITTER SETTINGS ###

train_proportion = 0.8
validation_proportion = 0.1
test_proportion = 0.1
rnd_seed = 42
spl_name = "splitter"

### IMAGE/MASK ENGINE SETTINGS ###

engine_settings_unet = {
    "image": {
        "do_rot_phase_avg": True,
        "do_resize": True,
        "resize_size": image_shape,
        "do_binarize": False,
    },
    "mask": {
        "do_rot_phase_avg": True,
        "do_resize": True,
        "resize_size": image_shape,
        "do_binarize": True,
        "binarize_func": "thresh",
    },
}
engine_settings_filtercnn = {
    "image": {
        "do_rot_phase_avg": True,
        "do_resize": True,
        "resize_size": image_shape,
        "do_binarize": False,
    },
    "mask": {
        "do_rot_phase_avg": True,
        "do_resize": True,
        "resize_size": image_shape,
        "do_binarize": True,
        "binarize_func": "thresh",
    },
    "mask_maker": {
        "model": "UNet",
        "trained_image_to_mask_network_path": "./models/trained_UNet_test_v0.pt",
    },
}
engine_settings_cnn1d = {
    "mask": {
        "do_rot_phase_avg": True,
        "do_resize": True,
        "resize_size": image_shape,
        "do_binarize": True,
        "binarize_func": "thresh",
    }
}

### TRAINER CONFIGURATION ###

config = {
    "generator": "simple",
    "batch_size": 10,
    "optimizer": "adam",
    "optim_lr": 0.001,
    # "loss"                    : 'cross_entropy',  #"cross_entropy"
    "save_best": True,
    "shuffle_train": True,
    # "shuffle_validation"      : True,
    # "shuffle_test"            : True,
    "device": "auto",
    "num_workers_dataloader": 0,
    # "pin_gpu_memory"          : True,
}
