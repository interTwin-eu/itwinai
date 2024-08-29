import psutil
import os
import time
import pandas as pd
import tensorflow as tf


class ProcessBenchmark(tf.keras.callbacks.Callback):
    """
    Gets several benchmarks parameters about the process during the training,
    including execution time.

    Parameters
    ----------
    filename : pathlike
        The file in which will be put the resulting history.

    """

    def __init__(self, filename):
        # get process pid
        self.pid = os.getpid()
        # instantiate a process object from which we'll get status information
        self.process = psutil.Process(self.pid)
        # get initial process cpu percent, so that the next call returns
        # the correct cpu percentage
        self.process.cpu_percent()

        self.filename = filename
        # create time history dataframe
        self.benchmark_df = pd.DataFrame(columns=[
            'time',
            'start_cpu_percent',
            'end_cpu_percent',
            'start_mem_rss',
            'end_mem_rss',
            'start_mem_vms',
            'end_mem_vms',
            'start_mem_uss',
            'end_mem_uss',
        ])
        # save the dataframe to csv file
        self.benchmark_df.to_csv(self.filename)

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time = -time.time()
        self.start_cpu_percent = self.process.cpu_percent()
        mem_info = self.process.memory_full_info()
        self.start_mem_rss = mem_info[0] / float(2 ** 20)
        self.start_mem_vms = mem_info[1] / float(2 ** 20)
        self.start_mem_uss = mem_info[3] / float(2 ** 20)

    def on_epoch_end(self, batch, logs={}):
        self.epoch_time += time.time()
        self.end_cpu_percent = self.process.cpu_percent()
        mem_info = self.process.memory_full_info()
        self.end_mem_rss = mem_info[0] / float(2 ** 20)
        self.end_mem_vms = mem_info[1] / float(2 ** 20)
        self.end_mem_uss = mem_info[3] / float(2 ** 20)

        self.benchmark_df.loc[len(self.benchmark_df.index)] = [
            self.epoch_time,
            self.start_cpu_percent,
            self.end_cpu_percent,
            self.start_mem_rss,
            self.end_mem_rss,
            self.start_mem_vms,
            self.end_mem_vms,
            self.start_mem_uss,
            self.end_mem_uss,
        ]

        self.benchmark_df.to_csv(self.filename)
