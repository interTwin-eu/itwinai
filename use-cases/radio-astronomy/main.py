from os import chdir
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# import src.pulsar_analysis as pa

from src.pulsar_analysis.neural_network_models import UNet, FilterCNN, CNN1D, WeightedBCELoss
from src.pulsar_analysis.pipeline_methods import PipelineImageToFilterDelGraphtoIsPulsar, PipelineImageToFilterToCCtoLabels
from pulsar_simulation.generate_data_pipeline import generate_example_payloads_for_training

# itwinai integration
from trainer import PulsarTrainer
from data import GenericDataset, CommonDataset, DatasetSplitter, SignalDataset, \
         pipelineLabelsInterface, pipelinePulsarInterface

# matplotlib.use('MacOSX')

## Generate synthetic data

generate_example_payloads_for_training(tag='train_v0_',
                                       num_payloads=10,
                                       plot_a_example=False,
                                       param_folder='./syn_payload/',
                                       payload_folder='./syn_payload/',
                                       num_cpus=1 #: choose based on the number of nodes/cores in your system
                                       )

generate_example_payloads_for_training(tag='test_v0_',
                                       num_payloads=50,
                                       plot_a_example=False,
                                       param_folder='./syn_data/runtime/',
                                       payload_folder='./syn_data/payloads/',
                                       num_cpus=10 #: choose based on the number of nodes/cores in your system
                                       )


mask_maker_dict = {
    "model": "UNet",
    "model_path": './models/trained_UNet_test_v0.pt',
}

image_tag='train_v0_*_payload_detected.json' #: '*' in the name is the index place holder of a image in the image set
image_directory='syn_payload/'

mask_tag = 'train_v0_*_payload_flux.json'
mask_directory='syn_payload/'

## Initialize the datasets
engine_settings_unet = {
    "image": {
        "do_rot_phase_avg": True,
        "do_resize": True,
        "resize_size": (128, 128),
        "do_binarize": False
    },
    "mask": {
        "do_rot_phase_avg": True,
        "do_resize": True,
        "resize_size": (128, 128),
        "do_binarize": True,
        "binarize_func": "thresh"
    }
}
engine_settings_filtercnn = {
    "image": {
        "do_rot_phase_avg": True,
        "do_resize": True,
        "resize_size": (128, 128),
        "do_binarize": False
    },
    "mask": {
        "do_rot_phase_avg": True,
        "do_resize": True,
        "resize_size": (128, 128),
        "do_binarize": True,
        "binarize_func": "thresh"
    },
    "mask_maker_engine": {
        "model": "UNet",
        "model_path": "./models/trained_UNet_test_v0.pt",
    }
}
engine_settings_cnn1d = {
    "mask": {
        "do_rot_phase_avg": True,
        "do_resize": True,
        "resize_size": (128, 128),
        "do_binarize": True,
        "binarize_func": "thresh"
    }
}

# ImgToMaskDs = GenericDataset(
#                     image_tag = image_tag,
#                     mask_tag= mask_tag,
#                     image_directory = image_directory,
#                     mask_directory = mask_directory,
#                     do_rot_phase_avg=True,
#                     do_resize=True,
#                     resize_size=(128,128), 
#                     mask_binarize_func="thresh"
#                     )

# InmaskToMaskDs = GenericDataset(
#                     image_tag = image_tag,
#                     mask_tag= mask_tag,
#                     image_directory = image_directory,
#                     mask_directory = mask_directory,
#                     mask_maker_engine=mask_maker_dict,
#                     do_rot_phase_avg=True,
#                     do_resize=True,
#                     resize_size=(128,128), 
#                     mask_binarize_func="thresh"
#                     )
# SignalToLabelDs = SignalDataset(
#                     mask_tag=mask_tag,
#                     mask_directory=mask_directory,
#                     do_rot_phase_avg=True,
#                     do_resize=True,
#                     resize_size=(128,128), 
#                     mask_binarize_func="thresh"
# )

ImgToMaskDs = CommonDataset(
                    image_tag = image_tag,
                    mask_tag= mask_tag,
                    image_directory = image_directory,
                    mask_directory = mask_directory,
                    type = 'unet',
                    engine_settings=engine_settings_unet,
                    )

InmaskToMaskDs = CommonDataset(
                    image_tag = image_tag,
                    mask_tag= mask_tag,
                    image_directory = image_directory,
                    mask_directory = mask_directory,
                    type = 'filtercnn',
                    engine_settings=engine_settings_filtercnn,
                    )
SignalToLabelDs = CommonDataset(
                    image_tag = image_tag,
                    mask_tag= mask_tag,
                    image_directory = image_directory,
                    mask_directory = mask_directory,
                    type = 'cnn1d',
                    engine_settings=engine_settings_cnn1d,
)


splitter = DatasetSplitter(
    train_proportion = 0.8,
    validation_proportion = 0.1,
    test_proportion = 0.1,
    rnd_seed = 42,
    name = 'splitter',
)


## Configuration for the trainers
config = {
    "generator": "simple",
    "batch_size": 10,
    "optimizer": "adam",
    "optim_lr": 0.001,
    # "loss": 'cross_entropy',  #"cross_entropy"
    "save_best": True,
    "shuffle_train": True,
    # "shuffle_validation": True,
    # "shuffle_test": True,
    "device": "auto",
    "num_workers_dataloader": 1,
    "pin_gpu_memory": True,
}

## Initialize the various trainers

ImgToMaskTrainer = PulsarTrainer(
                    model= UNet(),
                    epochs=2, #10
                    store_trained_model_at='./models/trained_UNet_test_v0.pt',
                    loss = WeightedBCELoss(pos_weight=3,neg_weight=1),
                    name='ImgToMaskTrainer',
                    config=config                               
                    )

InmaskToMaskTrainer = PulsarTrainer(
                    model= FilterCNN(),
                    epochs=3,
                    store_trained_model_at='./models/trained_Filter_test_v0.pt',
                    loss = WeightedBCELoss(pos_weight=1,neg_weight=1),  
                    name='InmaskToMaskTrainer',
                    config=config
                    )

SignalToLabelTrainer = PulsarTrainer(
                    model =CNN1D(),
                    epochs=20,
                    loss = WeightedBCELoss(pos_weight=1,neg_weight=1),
                    store_trained_model_at='./models/trained_CNN1D_test_v0.pt',
                    name='SignalToLabelTrainer',
                    config=config
                    )

ImgToMaskTrainer.execute(*splitter.execute(ImgToMaskDs))
InmaskToMaskTrainer.execute(*splitter.execute(InmaskToMaskDs))
SignalToLabelTrainer.execute(*splitter.execute(SignalToLabelDs))

ImgToMaskTrainer.write_model()
InmaskToMaskTrainer.write_model()
SignalToLabelTrainer.write_model()

# assemble pipeline and test on real data

ppl1f = PipelineImageToFilterDelGraphtoIsPulsar(
                    image_to_mask_network=UNet(),
                    trained_image_to_mask_network_path='./models/trained_UNet_test_v0.pt',
                    mask_filter_network=FilterCNN(),
                    trained_mask_filter_network_path='./models/trained_Filter_test_v0.pt',
                    signal_to_label_network=CNN1D(),
                    trained_signal_to_label_network='./models/trained_CNN1D_test_v0.pt'
                    )

ppl2f = PipelineImageToFilterToCCtoLabels(
                    image_to_mask_network=UNet(),
                    trained_image_to_mask_network_path='./models/trained_UNet_test_v0.pt',
                    mask_filter_network=FilterCNN(),
                    trained_mask_filter_network_path='./models/trained_Filter_test_v0.pt',
                    min_cc_size_threshold=5
                    )


image_directory_npy ='./test_data/joint_dataset_8_classes_real_remix_128x128.npy' 
label_directory_npy ='./test_data/joint_dataset_8_classes_real_remix_labels.npy' 
data = np.load(file=image_directory_npy,mmap_mode='r')
data_label = np.load(file=label_directory_npy,mmap_mode='r')
offset = 0
size_of_set = 500
data_subset = data[offset+1:offset+size_of_set,:,:]
data_label_subset = data_label[offset+1:offset+size_of_set]

while(True):
    ppl1f.test_on_real_data_from_npy_files(image_data_set=data_subset,image_label_set=data_label_subset,plot_details=True,plot_randomly=True,batch_size=2)
    continue_loop = not(input('[y/n] to quit: ')=='y')    
    plt.show()
    if continue_loop==False:
        break
    plt.close()
    ppl2f.test_on_real_data_from_npy_files(image_data_set=data_subset,image_label_set=data_label_subset,plot_randomly=True,batch_size=2)    
    continue_loop = not(input('[y/n] to quit: ')=='y')
    plt.show()
    if continue_loop==False:
        break
    plt.close()