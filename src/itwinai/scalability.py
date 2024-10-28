import glob
import os
import re
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_scalability_files(pattern_str: str, log_dir: Path, skip_id: int): 
    pattern = re.compile(pattern_str)
    csv_files = []

    for entry in log_dir.iterdir(): 
        match = pattern.search(str(entry))
        pass 

    for root, _, files in os.walk(os.getcwd()):
        for file in files:
            if not pattern.match(file):
                continue
            fpath = os.path.join(root, file)
            csv_files.append(fpath)
            df = pd.read_csv(fpath)
            if skip_id is not None:
                df = df.drop(df[df.epoch_id == skip_id].index)
            combined_df = pd.concat([combined_df, df])
    combined_df = pd.DataFrame(csv_files)
    return combined_df
