import re
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


def main(): 
    log_dir = Path("utilization_logs")
    pattern_str = r"dataframe_(?:\w+)_(?:\d+)\.csv$"
    pattern = re.compile(pattern_str)
    
    dataframes = []
    for entry in log_dir.iterdir():
        match = pattern.search(str(entry))
        if not match:
            continue
        print(f"Using data from file: {entry}")
        df = pd.read_csv(entry)
        dataframes.append(df)
    df = pd.concat(dataframes)
    # num_gpus = 8
    # csv_path = Path(f"utilization_logs/dataframe_{num_gpus}.csv")
    # df = pd.read_csv(csv_path)

    print(df.head(20))
    print()
    print(df["power"].mean())

    


if __name__ == "__main__": 
    main()
