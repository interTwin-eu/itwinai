from itwinai.components import DataGetter, DataSplitter, monitor_exec
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from pathlib import Path


class MNISTDataGetter(DataGetter):
    def __init__(
        self,
        data_root: str
    ) -> None:
        super().__init__()
        self.data_root = data_root

    @monitor_exec
    def execute(self):
        # Data
        print(self.data_root)
        self.data_root = Path(self.data_root).absolute()
        print(self.data_root)

        try:
            transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
            data = FashionMNIST(root=self.data_root, train=True,
                                download=False, transform=transform)
            print("Successfully loaded data!")
        except Exception as e:
            print(e)

        return data


class MNISTDataSplitter(DataSplitter):
    def __init__(
        self,
        train_proportion: int | float,
        validation_proportion: int | float = 0.0,
        test_proportion: int | float = 0.0,
    ) -> None:
        super().__init__(train_proportion, validation_proportion, test_proportion)

    @monitor_exec
    def execute(self, dataset):
        return dataset, None, None
