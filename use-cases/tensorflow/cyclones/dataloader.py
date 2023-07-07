import gdown
import argparse

# TODO: Wrap in Tensorflow components?


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='trainval.py', description="Train a VGG custom model on eFlows Dataset")

    # REQUIRED ARGUMENTS
    parser.add_argument("-o", "--output_dir", type=int, help="Output directory", required=True)
    args = parser.parse_args()

    url = 'https://drive.google.com/drive/folders/15DEq33MmtRvIpe2bNCg44lnfvEiHcPaf'
    gdown.download_folder(url=url, quiet=False)