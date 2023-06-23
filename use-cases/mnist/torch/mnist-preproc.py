import argparse
from torchvision.datasets import MNIST

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocessing of MNIST dataset.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path where to store preprocessed datasets.",
        default=None
    )
    # Syntactic sugar
    parser.add_argument(
        "-s", "--stage",
        type=str,
        help="Kind of dataset split to use.",
        default='train',
        choices=('train', 'test')
    )
    args = parser.parse_args()

    if args.stage == 'train':
        # Download and store training dataset
        MNIST(args.output, train=True, download=True)
    if args.stage == 'test':
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
