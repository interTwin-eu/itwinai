import lightning as L

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

class MNISTDataModule(L.LightningModule):
    def __init__(
        self,
        path: str,
        batch_size: int,
        train_prop: float,
    ) -> None:
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.train_prop = train_prop
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def setup(self, stage=None):
        if stage == "fit":
            mnist_full = MNIST(self.path, train=True, download=True, transform=self.transform)
            n_train_samples = int(self.train_prop * len(mnist_full))
            n_val_samples = len(mnist_full) - n_train_samples
            self.mnist_train, self.mnist_val = random_split(mnist_full, [n_train_samples, n_val_samples])

        if stage == "test":
            self.mnist_test = MNIST(self.path, train=False, download=True, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.path, train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=4)