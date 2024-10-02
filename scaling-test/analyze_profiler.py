import argparse
from pathlib import Path

import pandas as pd

def main(): 
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--filename",
        type=str,
        help="Path of the filename to read from",
    )
    args = parser.parse_args()

    df_path = Path(args.filename)
    df = pd.read_csv(df_path)
    df.sort_values("gpu_time", inplace=True, ascending=False)
    df.to_csv(df_path)

    print(df.head(20))
    


if __name__ == "__main__": 
    main()
