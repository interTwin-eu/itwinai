
import torch

from ._core import ModelDeviceHandler


if torch.cuda.is_available():
    torch_device = 'cuda'
else:
    torch_device = 'cpu'

torch.set_default_device(torch_device)
torch.set_default_dtype(torch.float64)  # double precision
