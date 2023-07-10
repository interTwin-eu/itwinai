import gdown
import argparse

# TODO: Wrap in Tensorflow components?


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # REQUIRED ARGUMENTS
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory", required=True)
    args = parser.parse_args()

    url = 'https://drive.google.com/drive/folders/15DEq33MmtRvIpe2bNCg44lnfvEiHcPaf'
    gdown.download_folder(url=url, quiet=False, output=args.output_dir)