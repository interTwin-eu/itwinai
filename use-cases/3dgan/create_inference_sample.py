"""Create a simple inference dataset sample and a checkpoint."""

import torch

CKPT_PATH = "3dgan-inference.pth"

if __name__ == "__main__":
    from model import ThreeDGAN
    net = ThreeDGAN()
    torch.save(net, CKPT_PATH)
