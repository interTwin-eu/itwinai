from .types import ModelML
import abc


class ModelLoader(abc.ABC):
    """Loads a machine learning model from somewhere."""

    def __init__(self, model_uri: str) -> None:
        super().__init__()
        self.model_uri = model_uri

    @abc.abstractmethod
    def __call__(self) -> ModelML:
        """Loads model from model URI."""
