import os
import pandas as pd

root_dir = '/p/scratch/intertwin/datasets/virgo'
output_file = './file_counts.txt'

with open(output_file, 'w') as f:
    for folder in range(1, 251):
        folder_path = os.path.join(root_dir, f'folder_{folder}')
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            df = pd.read_pickle(file_path)
            num_rows = df.shape[0]
            f.write(f'{file}: {num_rows}\n')
