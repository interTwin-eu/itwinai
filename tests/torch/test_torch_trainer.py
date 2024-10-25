import pytest

from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets

from itwinai.torch.trainer import TorchTrainer


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


@pytest.mark.hpc
def test_distributed_trainer():
    train_set = datasets.MNIST(
        '/p/project1/intertwin/smalldata/mnist', train=True, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    val_set = datasets.MNIST(
        '/p/project1/intertwin/smalldata/mnist', train=False, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    
    training_config = dict(
        optimizer='sgd',
        loss='nllloss'
    )
    trainer = TorchTrainer(
        model=Net(),
        config=training_config,
        epochs=2,
        strategy='ddp',
        checkpoint_every=1
    )
    trainer.execute(train_set, val_set)
