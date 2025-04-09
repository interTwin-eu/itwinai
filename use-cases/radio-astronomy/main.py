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
from data import PulsarDataset, DatasetSplitter, PipelineLabelsInterface, PipelinePulsarInterface

import inputs

if inputs.generate_data:
    generate_example_payloads_for_training( 
        tag             = inputs.tag,
        num_payloads    = inputs.num_payloads,
        plot_a_exampl   = inputs.plot_a_example,
        param_folder    = inputs.param_folder,
        payload_folder  = inputs.payload_folder,
        num_cpus        = inputs.num_cpus
    )

### INITIALIZE THE DATASETS ###

unet_ds = PulsarDataset(
    image_tag       = inputs.image_tag,
    mask_tag        = inputs.mask_tag,
    image_directory = inputs.image_directory,
    mask_directory  = inputs.mask_directory,
    type            = 'unet',
    engine_settings = inputs.engine_settings_unet
)

fcnn_ds = PulsarDataset(
    image_tag       = inputs.image_tag,
    mask_tag        = inputs.mask_tag,
    image_directory = inputs.image_directory,
    mask_directory  = inputs.mask_directory,
    type            = 'filtercnn',
    engine_settings = inputs.engine_settings_filtercnn
)

cnn1d_ds = PulsarDataset(
    image_tag       = inputs.image_tag,
    mask_tag        = inputs.mask_tag,
    image_directory = inputs.image_directory,
    mask_directory  = inputs.mask_directory,
    type            = 'cnn1d',
    engine_settings = inputs.engine_settings_cnn1d
)

### INITIALIZE THE SPLITTER ###

splitter = DatasetSplitter(
    train_proportion        = inputs.train_proportion,
    validation_proportion   = inputs.validation_proportion,
    test_proportion         = inputs.test_proportion,
    rnd_seed                = inputs.rnd_seed,
    name                    = inputs.spl_name
)

unet_trainer = PulsarTrainer(
    model                   = UNet(),
    epochs                  = 1,
    store_trained_model_at  = inputs.unet_dir,
    loss                    = WeightedBCELoss(pos_weight=3,neg_weight=1),
    name                    = 'UNet Trainer',
    config                  = inputs.config                               
)

fcnn_trainer = PulsarTrainer(
    model                   = FilterCNN(),
    epochs                  = 1,
    store_trained_model_at  = inputs.fcnn_dir,
    loss                    = WeightedBCELoss(pos_weight=1,neg_weight=1),  
    name                    = 'FilterCNN Trainer',
    config                  = inputs.config
)

cnn1d_trainer = PulsarTrainer(
    model                   = CNN1D(),
    epochs                  = 1,
    loss                    = WeightedBCELoss(pos_weight=1,neg_weight=1),
    store_trained_model_at  = inputs.cnn1d_dir,
    name                    = 'SignalToLabelTrainer',
    config                  = inputs.config
)

unet_trainer.execute(*splitter.execute(unet_ds))
fcnn_trainer.execute(*splitter.execute(fcnn_ds))
cnn1d_trainer.execute(*splitter.execute(cnn1d_ds))

unet_trainer.write_model()
fcnn_trainer.write_model()
cnn1d_trainer.write_model()

## assemble the full pipeline and test on real data

ppl1f = PipelineImageToFilterDelGraphtoIsPulsar(
    image_to_mask_network               = UNet(),
    trained_image_to_mask_network_path  = inputs.unet_dir,
    mask_filter_network                 = FilterCNN(),
    trained_mask_filter_network_path    = inputs.fcnn_dir,
    signal_to_label_network             = CNN1D(),
    trained_signal_to_label_network     = inputs.cnn1d_dir,
)

ppl2f = PipelineImageToFilterToCCtoLabels(
    image_to_mask_network               = UNet(),
    trained_image_to_mask_network_path  = inputs.unet_dir,
    mask_filter_network                 = FilterCNN(),
    trained_mask_filter_network_path    = inputs.fcnn_dir,
    min_cc_size_threshold               = 5
)


data                = np.load(file=inputs.image_directory_npy,mmap_mode='r')
data_label          = np.load(file=inputs.label_directory_npy,mmap_mode='r')
data_subset         = data[inputs.offset+1:inputs.offset+inputs.size_of_set,:,:]
data_label_subset   = data_label[inputs.offset+1:inputs.offset+inputs.size_of_set]

while(True):
    ppl1f.test_on_real_data_from_npy_files(image_data_set=data_subset,image_label_set=data_label_subset,plot_details=True,plot_randomly=True,batch_size=2)
    plt.show()
    if (input('[y/n] to quit: ')=='y'):
        break
    # continue_loop = not(input('[y/n] to quit: ')=='y')    
    # if continue_loop==False:
    #     break
    plt.close()
    ppl2f.test_on_real_data_from_npy_files(image_data_set=data_subset,image_label_set=data_label_subset,plot_randomly=True,batch_size=2)    
    plt.show()
    if (input('[y/n] to quit: ')=='y'):
        break
    # continue_loop = not(input('[y/n] to quit: ')=='y')
    # if continue_loop==False:
    #     break
    plt.close()