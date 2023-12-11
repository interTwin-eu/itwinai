from typing import Optional
from ..components import BaseComponent


class FakePreproc(BaseComponent):
    def __init__(self, max_items: int, name: Optional[str] = None
                 ) -> None:
        super().__init__(name)
        self.save_parameters(max_items=max_items, name=name)

    def execute(self):
        ...


class FakeTrainer(BaseComponent):
    def __init__(self, lr: float, batch_size: int, name: Optional[str] = None
                 ) -> None:
        super().__init__(name)
        self.save_parameters(lr=lr, batch_size=batch_size, name=name)

    def execute(self):
        ...


class FakeSaver(BaseComponent):
    def __init__(self, save_path: str, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.save_parameters(save_path=save_path, name=name)

    def execute(self):
        ...
