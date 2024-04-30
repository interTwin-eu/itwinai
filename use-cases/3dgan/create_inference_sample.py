"""Create a simple inference dataset sample and a checkpoint."""

import argparse
import os
import torch
from model import ThreeDGAN


def create_checkpoint(
    root: str = '.',
    ckpt_name: str = "3dgan-inference.pth"
):
    ckpt_path = os.path.join(root, ckpt_name)
    net = ThreeDGAN()
    torch.save(net, ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='.')
    parser.add_argument("--ckpt-name", type=str, default="3dgan-inference.pth")
    args = parser.parse_args()
    create_checkpoint(**vars(args))
