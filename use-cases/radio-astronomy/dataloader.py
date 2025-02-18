# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Oleksandr Krochak
# --------------------------------------------------------------------------------------

from itwinai.components import DataGetter, DataProcessor, monitor_exec
import os
from pulsar_simulation.generate_data_pipeline import generate_example_payloads_for_training
from typing import Optional

class synthesizeData(DataGetter):
    def __init__(self, name: Optional[str] = None,
                 tag: str = "test_v0_", num_payloads: int = 50, plot: bool = 0, num_cpus: int = 4, 
                 param_root: str = "syn_runtime/", payload_root: str = "syn_payload/") -> None:
       
        """Initialize the synthesizeData class.
    
        Args:
            name [optional] (str):  name of the data getter component.
            param_root      (str):  folder where synthetic param data will be saved.
            payload_root    (str):  folder where synthetic payload data will be saved.
            tag             (str):  tag which is used as prefix for the generated files.
            num_cpus        (int):  number of CPUs used for parallel processing.
            num_payloads    (int):  number of generated examples.
            plot            (bool): if True, plotting routine is activated \
                                               (set False when running 'main.py' directly after)
        """
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()), pop_self=False)

        # TODO find a smart way to compute the right value for num_cpus

        if not (os.path.exists(param_root) and os.path.exists(payload_root)):
            os.makedirs(param_root, exist_ok=True)
            os.makedirs(payload_root, exist_ok=True)

    @monitor_exec
    def execute(self) -> None:
        """Generate synthetic data and save it to disk. Relies on the pulsar_simulation package."""
        generate_example_payloads_for_training(tag         = self.parameters["tag"], 
                                            num_payloads   = self.parameters["num_payloads"],
                                            plot_a_example = self.parameters["plot"], 
                                            param_folder   = self.parameters["param_root"],
                                            payload_folder = self.parameters["payload_root"],
                                            num_cpus       = self.parameters["num_cpus"],
                                            reinit_ray     = False) 
                                            
# class normalizeData(DataProcessor):


testData = synthesizeData(num_payloads=10)
testData.execute()

trainData = synthesizeData(tag='train_v0_')
trainData.execute()
#     generate_example_payloads_for_training(tag='test_v0_',
#                                        num_payloads=50,
#                                        plot_a_example=False,
#                                        param_folder='./syn_data/runtime/',
#                                        payload_folder='./syn_data/payloads/',
#                                        num_cpus=1 #: choose based on the number of nodes/cores in your system
#                                        )
    




# class TimeSeriesDatasetGenerator(DataGetter):
#     def __init__(self, data_root: str = "data", name: Optional[str] = None) -> None:
#         """Initialize the TimeSeriesDatasetGenerator class.

#         Args:
#             data_root (str): Root folder where datasets will be saved.
#             name (Optional[str]): Name of the data getter component.
#         """
#         super().__init__(name)
#         self.save_parameters(**self.locals2params(locals()))
#         self.data_root = data_root
#         if not os.path.exists(data_root):
#             os.makedirs(data_root, exist_ok=True)

#     @monitor_exec
#     def execute(self) -> pd.DataFrame:
#         """Generate a time-series dataset, convert it to Q-plots,
#         save it to disk, and return it.

#         Returns:
#             pd.DataFrame: dataset of Q-plot images.
#         """
#         df_aux_ts = generate_dataset_aux_channels(
#             1000,
#             3,
#             duration=16,
#             sample_rate=500,
#             num_waves_range=(20, 25),
#             noise_amplitude=0.6,
#         )
#         df_main_ts = generate_dataset_main_channel(
#             df_aux_ts, weights=None, noise_amplitude=0.1
#         )

#         # save datasets
#         save_name_main = "TimeSeries_dataset_synthetic_main.pkl"
#         save_name_aux = "TimeSeries_dataset_synthetic_aux.pkl"
#         df_main_ts.to_pickle(os.path.join(self.data_root, save_name_main))
#         df_aux_ts.to_pickle(os.path.join(self.data_root, save_name_aux))

#         # Transform to images and save to disk
#         df_ts = pd.concat([df_main_ts, df_aux_ts], axis=1)
#         df = generate_cut_image_dataset(
#             df_ts, list(df_ts.columns), num_processes=20, square_size=64
#         )
#         save_name = "Image_dataset_synthetic_64x64.pkl"
#         df.to_pickle(os.path.join(self.data_root, save_name))
#         return df