# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from typing import Optional

from ..components import BaseComponent, monitor_exec


class FakeGetter(BaseComponent):
    def __init__(self, data_uri: str, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.save_parameters(data_uri=data_uri, name=name)
        self.data_uri = data_uri

    def execute(self): ...


class FakeGetterExec(FakeGetter):
    result: str = "dataset"

    @monitor_exec
    def execute(self):
        return self.result


class FakeSplitter(BaseComponent):
    def __init__(self, train_prop: float, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.save_parameters(train_prop=train_prop, name=name)
        self.train_prop = train_prop

    def execute(self): ...


class FakeSplitterExec(FakeSplitter):
    result: tuple = ("train_dataset", "val_dataset", "test_dataset")

    @monitor_exec
    def execute(self, dataset):
        return self.result


class FakePreproc(BaseComponent):
    def __init__(self, max_items: int, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.save_parameters(max_items=max_items, name=name)
        self.max_items = max_items

    @monitor_exec
    def execute(self): ...


class FakePreprocExec(FakePreproc):
    @monitor_exec
    def execute(self, train_dataset, val_dataset, test_dataset):
        return train_dataset, val_dataset, test_dataset


class FakeTrainer(BaseComponent):
    def __init__(self, lr: float, batch_size: int, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.save_parameters(lr=lr, batch_size=batch_size, name=name)
        self.lr = lr
        self.batch_size = batch_size

    @monitor_exec
    def execute(self): ...


class FakeTrainerExec(FakeTrainer):
    model: str = "trained_model"

    @monitor_exec
    def execute(self, train_dataset, val_dataset, test_dataset):
        return train_dataset, val_dataset, test_dataset, self.model


class FakeSaver(BaseComponent):
    def __init__(self, save_path: str, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.save_parameters(save_path=save_path, name=name)
        self.save_path = save_path

    def execute(self): ...


class FakeSaverExec(FakeSaver):
    @monitor_exec
    def execute(self, artifact):
        return artifact
