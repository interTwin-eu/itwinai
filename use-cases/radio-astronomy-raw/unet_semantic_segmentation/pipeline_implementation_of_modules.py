from os import chdir
import matplotlib.pyplot as plt
# %matplotlib inline
# chdir('../')
from pulsar_simulation.generate_data_pipeline import generate_example_payloads_for_training

from src.pulsar_analysis.train_neural_network_model import \
        ImageMaskPair, TrainImageToMaskNetworkModel, ImageToMaskDataset, \
        TrainSignalToLabelModel,SignalToLabelDataset

from src.pulsar_analysis.preprocessing import PrepareFreqTimeImage, BinarizeToMask

from src.pulsar_analysis.postprocessing import DelayGraph,LineClassifier,ConnectedComponents,FitSegmentedTraces

from src.pulsar_analysis.neural_network_models import \
        UNet, CustomLossUNet, CNN1D, FilterCNN, OneDconvEncoder

from src.pulsar_analysis.information_packet_formats import Payload

from src.pulsar_analysis.pipeline_methods import \
        ImageDataSet, ImageReader, PipelineImageToDelGraphtoIsPulsar, \
        LabelDataSet, LabelReader, PipelineImageToCCtoLabels

generate_example_payloads_for_training(tag='train_v0_',
                                       num_payloads=500,
                                       plot_a_example=True,
                                       param_folder='./syn_data/runtime/',
                                       payload_folder='./syn_data/payloads/',
                                       num_cpus=10 #: choose based on the number of nodes/cores in your system
                                       )

generate_example_payloads_for_training(tag='test_v0_',
                                       num_payloads=500,
                                       plot_a_example=True,
                                       param_folder='./syn_data/runtime/',
                                       payload_folder='./syn_data/payloads/',
                                       num_cpus=10 #: choose based on the number of nodes/cores in your system
                                       )



#: Load a Freq-Time Image and its segmented pair as ImageMaskPair object
image_payload_address = './syn_data/payloads/test_v0_400_payload_detected.json'
mask_payload_address = './syn_data/payloads/test_v0_400_payload_flux.json'

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
                                                binarize_engine=BinarizeToMask(binarize_func='gaussian_blur') # or 'exponential'
                                                )

im_obj = ImageMaskPair.load_from_payload_address(
                                                image_payload_address=image_payload_address,
                                                mask_payload_address=mask_payload_address,
                                                image_engine=image_preprocessing_engine,
                                                mask_engine=mask_preprocessing_engine)

#: Visualize 
im_obj.plot()
# or retrieve as tuple of tensors as im_obj()
print(im_obj.descriptions[0])


# Retrive the image and mask and calculate delay graph from mask (or image)
image,mask = im_obj()
delay_graph_engine = DelayGraph(normalize_delays=True)
x_lags,y_pos = delay_graph_engine(dispersed_freq_time=mask.detach().numpy())

#: Define a Line classifier to detect possibility of a Pulse
LineClassifier_obj = LineClassifier(no_pulsar_slope_range=[87,93])
LineClassifier_obj.plot(x_lags_normalized=x_lags,y_channels_normalized=y_pos)
decision = LineClassifier_obj(x_lags_normalized=x_lags,y_channels_normalized=y_pos)
print(f'Decision about presence of pulsar is {decision}')


#: Instantiate a Connected Component engine
cc_obj = ConnectedComponents(small_component_size=10)
cc_obj.plot(dispersed_freq_time_segmented=mask.detach().numpy())

labelled_skeleton = cc_obj(dispersed_freq_time_segmented=mask.detach().numpy())
FitSegmentedTraces.fitt_to_all_traces(labelled_skeleton=labelled_skeleton)
FitSegmentedTraces.plot_all_traces(labelled_skeleton=labelled_skeleton)



image_tag='train_v0_*_payload_detected.json' #: '*' in the name is the index place holder of a image in the image set
mask_tag='train_v0_*_payload_flux.json'
image_directory='./syn_data/payloads/'
mask_directory='./syn_data/payloads/'
store_trained_model_at: str = './syn_data/model/trained_UNet_test_v0.pt'
store_trained_sig_label_model_at: str = './syn_data/model/trained_CNN1D_test_v0.pt'


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
                                                binarize_engine=BinarizeToMask(binarize_func='gaussian_blur') # or 'exponential'
                                                )

train_data_set = ImageToMaskDataset(
                        image_tag = image_tag,
                        mask_tag= mask_tag,
                        image_directory = image_directory,
                        mask_directory = mask_directory,
                        image_engine=image_preprocessing_engine,
                        mask_engine=mask_preprocessing_engine
                        )
train_data_set.plot(index=200)

signal_label_dataset = SignalToLabelDataset(mask_tag=mask_tag,
                                            mask_directory=mask_directory,
                                            mask_engine=mask_preprocessing_engine,
                                            )


#: Instantiate
trainer = TrainImageToMaskNetworkModel(
                                model=UNet(),
                                num_epochs=5,
                                store_trained_model_at=store_trained_model_at,
                                loss_criterion = CustomLossUNet()                                
                                )

#: Start training  
#trainer(image_mask_pairset=train_data_set)


trainer_sig_label = TrainSignalToLabelModel(
                                model=CNN1D(),
                                num_epochs=10,
                                store_trained_model_at=store_trained_sig_label_model_at,                                                                
                                )

#: Start training  
#trainer_sig_label(signal_label_pairset=signal_label_dataset)


#: Instantiate the test data set like train data set
image_tag='test_v0_*_payload_detected.json' #: '*' in the name is the index place holder of a image in the image set
mask_tag='test_v0_*_payload_flux.json'
image_directory='./syn_data/payloads/'
mask_directory='./syn_data/payloads/'

test_data_set = ImageToMaskDataset(
                        image_tag = image_tag,
                        mask_tag= mask_tag,
                        image_directory = image_directory,
                        mask_directory = mask_directory,
                        image_engine=image_preprocessing_engine,
                        mask_engine=mask_preprocessing_engine
                        )
test_data_set.plot(index=4)

test_signal_label_dataset = SignalToLabelDataset(mask_tag=mask_tag,
                                            mask_directory=mask_directory,
                                            mask_engine=mask_preprocessing_engine,
                                            )


id = 100
test_data_set.plot(index=id)
image = test_data_set.__getitem__(index=id)[0]
mask = test_data_set.__getitem__(index=id)[1]
signal = test_signal_label_dataset.__getitem__(index=id)[0]

#print(test_data_set.__get_descriptions__(index=id)[0])
#print(signal)
pred = trainer.test_model(image=image,plot_pred=True)
print(pred.shape,mask.detach().numpy().shape)
pred_labels_mask = trainer_sig_label.test_model(mask=mask.squeeze().detach().numpy())
pred_labels = trainer_sig_label.test_model(mask=pred)
expected_labels = test_signal_label_dataset.__getitem__(index=id)[1]
print(f'expected: {expected_labels} and predicted from pred,mask {pred_labels,pred_labels_mask}')

labelled_skeleton = cc_obj(dispersed_freq_time_segmented=pred)
#FitSegmentedTraces.fitt_to_all_traces(labelled_skeleton=labelled_skeleton)
FitSegmentedTraces.plot_all_traces_with_categories(labelled_skeleton=labelled_skeleton,image=image.squeeze())


id = 168

image_tag='test_v0_*_payload_detected.json' #: '*' in the name is the index place holder of a image in the image set
image_directory='syn_data/payloads/'
im_set = ImageDataSet(image_tag=image_tag,image_directory=image_directory,image_reader_engine=ImageReader(file_type=Payload([]),do_average=False))
label_set = LabelDataSet(image_tag=image_tag,image_directory=image_directory,label_reader_engine=LabelReader(file_type=Payload([])))
image = im_set.__getitem__(idx=id)
im_set.plot(idx=id)
ppl1 = PipelineImageToDelGraphtoIsPulsar(image_to_mask_network=UNet(),
                                        trained_image_to_mask_network_path=store_trained_model_at,
                                        signal_to_label_network=CNN1D(),
                                        trained_signal_to_label_network=store_trained_sig_label_model_at)
label = ppl1(image=image)
print(f'is pulsar present? {label}')

#ppl1.validate_efficiency(image_data_set=im_set,label_data_set=label_set)

id = 157

image_tag='test_v0_*_payload_detected.json' #: '*' in the name is the index place holder of a image in the image set
image_directory='syn_data/payloads/'
im_set = ImageDataSet(image_tag=image_tag,image_directory=image_directory,image_reader_engine=ImageReader(file_type=Payload([]),do_average=False))
label_set = LabelDataSet(image_tag=image_tag,image_directory=image_directory,label_reader_engine=LabelReader(file_type=Payload([])))
image = im_set[id]
im_set.plot(idx=id)
ppl2 = PipelineImageToCCtoLabels(image_to_mask_network=UNet(),
                                trained_image_to_mask_network_path=store_trained_model_at,
                                )
label = ppl2(image=image)
print(f'[pulsar, NBRI, BBRFI, None] scores: {list(label.values())}')

#ppl2.validate_efficiency(image_data_set=im_set,label_data_set=label_set)

import numpy as np
image_directory_npy ='path_to_real_image_data' #: load numpy memmap array containing real pulsar dispersion graphs. If not then design your own dataloader class 
label_directory_npy ='path_to_real_label_data' #: load numpy  array containing corrsponding label. If not then design your own dataloader class 
data = np.load(file=image_directory_npy,mmap_mode='r')
data_label = np.load(file=label_directory_npy,mmap_mode='r')
#data[0,:,:]
id = 39
is_pulsar = ppl1(image=data[id,:,:],return_bool=True)
is_pulsar_cc = ppl2(image=data[id,:,:])
is_pulsar_there = data_label[id]
print(f'Label:{is_pulsar_there}  is_pulsar {is_pulsar} and cc {is_pulsar_cc}')


offset = 500
size_of_set = 500
data_subset = data[offset+1:offset+size_of_set,:,:]
data_label_subset = data_label[offset+1:offset+size_of_set]

#type(data_label_subset)==np.memmap
ppl1.test_on_real_data_from_npy_files(image_data_set=data_subset,image_label_set=data_label_subset,plot_details=True,plot_randomly=True,batch_size=2)

#ppl1.test_on_real_data_from_npy_files(image_data_set=data_subset)
#np.random.permutation(np.arange(data_subset.shape[0]))[0:5]