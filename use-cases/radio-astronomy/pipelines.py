from src.pulsar_analysis.pipeline_methods import PipelineImageToFilterDelGraphtoIsPulsar, PipelineImageToFilterToCCtoLabels
import numpy as np

class pipelinePulsarInterface(PipelineImageToFilterDelGraphtoIsPulsar):
    def execute(self) -> PipelineImageToFilterDelGraphtoIsPulsar:
        return self
    
    # interface method
    def test_on_real_data_from_npy_files(self, img_dir, lbl_dir, offset, size):
        data = np.load(file=img_dir,mmap_mode='r')
        data_label = np.load(file=lbl_dir,mmap_mode='r')
        data_subset = data[offset+1:offset+size,:,:]
        data_label_subset = data_label[offset+1:offset+size]
        
        return super().test_on_real_data_from_npy_files(
            image_data_set=data_subset,
            image_label_set=data_label_subset,
            plot_details=True,
            plot_randomly=True,
            batch_size=2
        )


class pipelineLabelsInterface(PipelineImageToFilterToCCtoLabels):
    def execute(self) -> PipelineImageToFilterToCCtoLabels:
        return self

    # interface method
    def test_on_real_data_from_npy_files(self, img_dir, lbl_dir, offset, size):
        data = np.load(file=img_dir,mmap_mode='r')
        data_label = np.load(file=lbl_dir,mmap_mode='r')
        data_subset = data[offset+1:offset+size,:,:]
        data_label_subset = data_label[offset+1:offset+size]
        
        return super().test_on_real_data_from_npy_files(
            image_data_set=data_subset,
            image_label_set=data_label_subset,
            plot_details=True,
            plot_randomly=True,
            batch_size=2
        )
