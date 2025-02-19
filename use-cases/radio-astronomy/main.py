from os import chdir
import numpy as np
import matplotlib.pyplot as plt

from src.pulsar_analysis.train_neural_network_model import ImageMaskPair
from src.pulsar_analysis.preprocessing import PrepareFreqTimeImage, BinarizeToMask
from src.pulsar_analysis.postprocessing import DelayGraph,LineClassifier,ConnectedComponents,FitSegmentedTraces

from src.pulsar_analysis.train_neural_network_model import TrainImageToMaskNetworkModel,ImageToMaskDataset,InMaskToMaskDataset
from src.pulsar_analysis.neural_network_models import UNet, CustomLossUNet, UNetFilter, FilterCNN, CustomLossSemanticSeg, CNN1D, WeightedBCELoss

from src.pulsar_analysis.train_neural_network_model import TrainSignalToLabelModel,SignalToLabelDataset
from src.pulsar_analysis.neural_network_models import OneDconvEncoder,Simple1DCnnClassifier

from src.pulsar_analysis.pipeline_methods import ImageDataSet, ImageReader,LabelDataSet,LabelReader,PipelineImageToCCtoLabels, PipelineImageToMask, PipelineImageToFilterToCCtoLabels

from src.pulsar_analysis.information_packet_formats import Payload
from src.pulsar_analysis.pipeline_methods import ImageDataSet, ImageReader, PipelineImageToDelGraphtoIsPulsar,PipelineImageToFilterDelGraphtoIsPulsar,LabelDataSet,LabelReader

# Alex

from trainer import PulsarTrainer


image_preprocessing_engine = PrepareFreqTimeImage(
                                                do_rot_phase_avg=True,
                                                do_binarize=False,
                                                do_resize=True,
                                                resize_size=(128,128),
                                                )
mask_preprocessing_engine = PrepareFreqTimeImage(
                                                do_rot_phase_avg=True,
                                                do_binarize=True,
                                                do_resize=True,
                                                resize_size=(128,128),
                                                binarize_engine = BinarizeToMask(binarize_func="thresh")#BinarizeToMask(binarize_func='gaussian_blur') # or 'exponential'
                                                )

cnn_model_to_make_mask_path: str = './trained_UNet_test_v0.pt'
mask_maker_engine = PipelineImageToMask(
                                image_to_mask_network=UNet(),
                                trained_image_to_mask_network_path=cnn_model_to_make_mask_path,                     
                                )

signal_maker_engine = DelayGraph()

label_reader_engine = LabelReader()

image_tag='train_v0_*_payload_detected.json' #: '*' in the name is the index place holder of a image in the image set
image_directory='syn_payload/'

mask_tag = 'train_v0_*_payload_flux.json'
mask_directory='syn_payload/'

store_trained_model_image2mask_at = './models/trained_UNet_test_v0.pt'


image_mask_train_dataset = ImageToMaskDataset(
                        image_tag = image_tag,
                        mask_tag= mask_tag,
                        image_directory = image_directory,
                        mask_directory = mask_directory,
                        image_engine=image_preprocessing_engine,
                        mask_engine=mask_preprocessing_engine
                        )


### PDDT

# image2mask_network_trainer = TrainImageToMaskNetworkModel(
#                                     model=UNet(),
#                                     num_epochs=2, #10
#                                     store_trained_model_at=store_trained_model_image2mask_at,
#                                     loss_criterion = WeightedBCELoss(pos_weight=3,neg_weight=1)                                
#                                     )



# image2mask_network_trainer(image_mask_pairset=image_mask_train_dataset)

### MYWAY

config = {
    "generator": "simple",
    "batch_size": 1,
    "optimizer": "adam",
    "optim_lr": 0.001,
    "loss": 'cross_entropy',
    "save_best": True,
    "shuffle_train": True,
    "device": "auto"
}


image2mask_network_trainer = PulsarTrainer(
                                    model=UNet(),
                                    num_epochs=2, #10
                                    store_trained_model_at=store_trained_model_image2mask_at,
                                    config=config                               
                                    )

# if input('Type [y/n] to train') == 'y':



image2mask_network_trainer.create_dataloaders(train_dataset=image_mask_train_dataset)
image2mask_network_trainer.train()
# store_trained_model_inmask2mask_at = './models/trained_FilterCNN_test_v0.pt'
# inmask2mask_network_trainer = TrainImageToMaskNetworkModel(
#                                     model= FilterCNN(),
#                                     num_epochs=3,
#                                     store_trained_model_at=store_trained_model_inmask2mask_at,
#                                     loss_criterion = WeightedBCELoss(pos_weight=1,neg_weight=1)                               
#                                     )

# store_trained_model_signal2label_at: str = './models/trained_CNN1D_test_v0.pt'
# signal2label_network_trainer = TrainSignalToLabelModel(
#                                     model=CNN1D(),
#                                     num_epochs=20,
#                                     loss_criterion=WeightedBCELoss(pos_weight=1,neg_weight=1),
#                                     store_trained_model_at=store_trained_model_signal2label_at,                                                                
#                                     )

# inmask_mask_train_dataset = InMaskToMaskDataset(
#                         image_tag = image_tag,
#                         mask_tag= mask_tag,
#                         image_directory = image_directory,
#                         mask_directory = mask_directory,
#                         mask_maker_engine=mask_maker_engine,
#                         image_engine=image_preprocessing_engine,
#                         mask_engine=mask_preprocessing_engine
#                         )

# inmask2mask_network_trainer(image_mask_pairset=inmask_mask_train_dataset)

# signal_label_train_dataset = SignalToLabelDataset(mask_tag=mask_tag,
#                                             mask_directory=mask_directory,
#                                             mask_engine=mask_preprocessing_engine,
#                                             )

# signal2label_network_trainer(signal_label_pairset=signal_label_train_dataset)


# ppl1f = PipelineImageToFilterDelGraphtoIsPulsar(image_to_mask_network=UNet(),
#                                         trained_image_to_mask_network_path=store_trained_model_image2mask_at,
#                                         mask_filter_network=FilterCNN(),
#                                         trained_mask_filter_network_path=store_trained_model_inmask2mask_at,
#                                         signal_to_label_network=CNN1D(),
#                                         trained_signal_to_label_network=store_trained_model_signal2label_at)

# ppl2f = PipelineImageToFilterToCCtoLabels(image_to_mask_network=UNet(),
#                                 trained_image_to_mask_network_path=store_trained_model_image2mask_at,
#                                 mask_filter_network=FilterCNN(),
#                                 trained_mask_filter_network_path=store_trained_model_inmask2mask_at,
#                                 min_cc_size_threshold=5)


# image_directory_npy ='./test_data_example/joint_dataset_8_classes_real_remix_128x128.npy' 
# label_directory_npy ='./test_data_example/joint_dataset_8_classes_real_remix_labels.npy' 
# data = np.load(file=image_directory_npy,mmap_mode='r')
# data_label = np.load(file=label_directory_npy,mmap_mode='r')
# offset = 0
# size_of_set = 500
# data_subset = data[offset+1:offset+size_of_set,:,:]
# data_label_subset = data_label[offset+1:offset+size_of_set]

# while(True):
#     ppl1f.test_on_real_data_from_npy_files(image_data_set=data_subset,image_label_set=data_label_subset,plot_details=True,plot_randomly=True,batch_size=2)
#     continue_loop = not(input('[y/n] to quit: ')=='y')    
#     # plt.show()
#     if continue_loop==False:
#         break
#     # plt.close()
#     ppl2f.test_on_real_data_from_npy_files(image_data_set=data_subset,image_label_set=data_label_subset,plot_randomly=True,batch_size=2)    
#     continue_loop = not(input('[y/n] to quit: ')=='y')
#     # plt.show()
#     if continue_loop==False:
#         break
#     # plt.close()

    
    