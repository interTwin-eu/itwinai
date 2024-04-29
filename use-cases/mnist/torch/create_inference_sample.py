"""Create a simple inference dataset sample and a checkpoint."""

import torch
import os

from model import Net
from dataloader import InferenceMNIST


def mnist_torch_inference_files(
    root: str = '.',
    samples_path: str = 'mnist-sample-data/',
    model_name: str = 'mnist-pre-trained.pth'
):
    """Create sample dataset and fake model to test mnist
    inference workflow. Assumes to be run from
    the use case folder.

    Args:
        root (str, optional): where to create the files.
        Defaults to '.'.
    """

    sample = os.path.join(root, samples_path)
    InferenceMNIST.generate_jpg_sample(sample, 10)

    # Fake checkpoint
    dummy_nn = Net()
    mdl_ckpt = os.path.join(root, model_name)
    torch.save(dummy_nn, mdl_ckpt)


if __name__ == "__main__":
    mnist_torch_inference_files()
