from src.pulsar_analysis.pipeline_methods import PipelineImageToFilterDelGraphtoIsPulsar, PipelineImageToFilterToCCtoLabels
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

class pipelinePulsarInterface(PipelineImageToFilterDelGraphtoIsPulsar):
    def execute(self) -> PipelineImageToFilterDelGraphtoIsPulsar:
        return self
    
class pipelineLabelsInterface(PipelineImageToFilterToCCtoLabels):
    def execute(self) -> PipelineImageToFilterToCCtoLabels:
        return self

class testSuite:
    def __init__(
        self,
        image_to_mask_network: nn.Module,
        trained_image_to_mask_network_path: str,
        mask_filter_network: nn.Module,
        trained_mask_filter_network_path: str,
        signal_to_label_network: nn.Module,
        trained_signal_to_label_network: str,
        img_dir: str,
        lbl_dir: str,
        size: int,
        offset: int,
        ):
            self.img_dir = img_dir
            self.lbl_dir = lbl_dir
            self.size = size
            self.offset = offset

            self.DelGraphtoIsPulsar = PipelineImageToFilterDelGraphtoIsPulsar(
                image_to_mask_network,
                trained_image_to_mask_network_path,
                mask_filter_network,
                trained_mask_filter_network_path,
                signal_to_label_network,
                trained_signal_to_label_network   
            )

            self.ToCCtoLabels = PipelineImageToFilterToCCtoLabels(
                image_to_mask_network,
                trained_image_to_mask_network_path,
                mask_filter_network,
                trained_mask_filter_network_path,
                min_cc_size_threshold=5
            )

    def execute(self):
        data = np.load(file=self.img_dir,mmap_mode='r')
        data_label = np.load(file=self.lbl_dir,mmap_mode='r')
        data_subset = data[self.offset+1:self.offset+self.size,:,:]
        data_label_subset = data_label[self.offset+1:self.offset+self.size]

        self.DelGraphtoIsPulsar.test_on_real_data_from_npy_files(
            image_data_set=data_subset,
            image_label_set=data_label_subset,
            plot_details=True,
            plot_randomly=True,
            batch_size=2
        )

        self.ToCCtoLabels.test_on_real_data_from_npy_files(
            image_data_set=data_subset,
            image_label_set=data_label_subset,
            plot_randomly=True,
            batch_size=2
        )

        plt.show()
        return print("Test Suite executed")       
        
