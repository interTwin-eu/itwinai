from os import chdir
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# import src.pulsar_analysis as pa

from src.pulsar_analysis.neural_network_models import UNet, FilterCNN, CNN1D, WeightedBCELoss
from src.pulsar_analysis.pipeline_methods import PipelineImageToFilterDelGraphtoIsPulsar, PipelineImageToFilterToCCtoLabels

matplotlib.use('MacOSX')

# itwinai integration
from trainer import PulsarTrainer
from data import GenericDataset, DatasetSplitter, SignalDataset, SynthesizeData

dataGen = SynthesizeData(
    tag = "train_v0_", 
    num_payloads = 10)

dataGen.execute()
mask_maker_dict = {
    "model": "UNet",
    "model_path": './models/trained_UNet_test_v0.pt',
}

image_tag='train_v0_*_payload_detected.json' #: '*' in the name is the index place holder of a image in the image set
image_directory='syn_payload/'

mask_tag = 'train_v0_*_payload_flux.json'
mask_directory='syn_payload/'

## Initialize the datasets

ImgToMaskDs = GenericDataset(
                    image_tag = image_tag,
                    mask_tag= mask_tag,
                    image_directory = image_directory,
                    mask_directory = mask_directory,
                    do_rot_phase_avg=True,
                    do_resize=True,
                    resize_size=(128,128), 
                    mask_binarize_func="thresh"
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
                    num_epochs=10, #10
                    store_trained_model_at='./models/trained_UNet_test_v0.pt',
                    loss = WeightedBCELoss(pos_weight=3,neg_weight=1),
                    name='ImgToMaskTrainer',
                    config=config                               
                    )

ImgToMaskTrainer.execute(*splitter.execute(ImgToMaskDs))


ImgToMaskTrainer.write_model()
