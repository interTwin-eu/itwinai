"""
Test Trainer class. To run this script, use the following command:

>>> torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29400 test_trainer.py

"""

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from itwinai.backend.torch.trainer import TorchTrainer2


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)


if __name__ == '__main__':
    train_set = datasets.MNIST(
        '.tmp/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    val_set = datasets.MNIST(
        '.tmp/', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    trainer = TorchTrainer2(
        model=Net(),
        train_dataloader=DataLoader(train_set, batch_size=32, pin_memory=True),
        validation_dataloader=DataLoader(
            val_set, batch_size=32, pin_memory=True),
        strategy='ddp',
        backend='nccl',
        loss='NLLLoss',
        epochs=20,
        checkpoint_every=1
    )
    trainer.execute()
