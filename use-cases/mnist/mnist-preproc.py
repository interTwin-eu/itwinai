import argparse
from torchvision.datasets import MNIST

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing of MNIST dataset.')
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Path to input dataset.',
        default=None
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path to output dataset (preprocessed dataset).',
        default=None
    )
    args = parser.parse_args()

    MNIST(args.output, train=True, download=True)
    MNIST(args.output, train=False, download=True)

    print(
        """
    ******************************
    * Called MNIST preprocessing *
    ******************************

    - Download dataset
    - TODO: Split the dataset in training and inference
    - TODO: Preprocess it
    - Store it to local filesystem

    """
    )
