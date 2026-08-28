import subprocess
from pathlib import Path

from .base import BaseBackend


# TODO: Can be extended to add other backends, e.g. HuggingFace
class AIModelHubBackend(BaseBackend):
    def upload(self, model_dir: Path):
        subprocess.run(
            ["itwinai", "upload-model-to-hub", str(model_dir)],
            check=False,
        )
