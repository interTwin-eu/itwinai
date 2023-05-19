import argparse
from torchvision.datasets import MNIST

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocessing of MNIST dataset.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to output dataset (preprocessed dataset).",
        default=None
    )
    args = parser.parse_args()

    # Download and store training dataset
    MNIST(args.output, train=True, download=True)
    # Download and store test dataset
    MNIST(args.output, train=False, download=True)

    print(
        """
    ******************************
    * Called MNIST preprocessing *
    ******************************

    - Download dataset
    - Split the dataset in training and inference
    - Preprocess it
    - Store it to local filesystem

    """
    )
