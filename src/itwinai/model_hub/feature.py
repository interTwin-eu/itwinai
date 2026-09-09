from pathlib import Path

import torch
from torch import nn

from ..serialization import ModelLoader
from .backends import get_backend
from .download import discover_weights_file, download_file
from .manifest import write_manifest


class ModelHubFeature:
    def __init__(self, config: dict):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.final_checkpoint_name = self.config.get("final_checkpoint_name", "best_model")

        backend_name = self.config.get("backend", "ai-model-hub")
        self.backend = get_backend(backend_name, self.config)

    def on_checkpoint_saved(self, trainer, ckpt_dir):
        if not self.enabled:
            return
        ckpt_dir = Path(ckpt_dir)
        write_manifest(ckpt_dir, self.config)

    def on_training_end(self, trainer, ckpt_dir):
        if not self.enabled:
            return
        ckpt_dir = Path(ckpt_dir)
        write_manifest(ckpt_dir, self.config)

        mode = self.config.get("mode", "deferred")
        if mode == "online":
            self._safe_upload(ckpt_dir)
        elif mode == "auto":
            if has_internet_connection():
                self._safe_upload(ckpt_dir)
            else:
                print(f"Model Hub config ready in: {ckpt_dir}")
        elif mode == "deferred":
            print(f"Model Hub can be run in: {ckpt_dir}")

    def _safe_upload(self, ckpt_dir: Path) -> None:
        try:
            self.backend.upload(ckpt_dir)
        except Exception as e:
            print(f"Model Hub upload failed for checkpoint at '{ckpt_dir}': {e}")
            print(f"You can re-upload later with: itwinai upload-model-to-hub {ckpt_dir}")


class ModelHubModelLoader(ModelLoader):
    """Pulls a model file from the RI-SCALE Model Hub and loads it as a
    torch model.

    Args:
        model_id (str): Model Hub artifact ID.
        file_path (str | None): Name of the file to download within the model's
            artifact. If None, will attempt to auto-discover the weights file.
        model_class (nn.Module | None): required if the checkpoint is
            a state dict rather than a full pickled model.
        base_url (str): Model Hub artifacts base URL.
    """

    def __init__(
        self,
        model_id: str,
        file_path: str | None = None,
        model_class: nn.Module | None = None,
        base_url: str = "https://hypha.aicell.io/ri-scale/artifacts",
    ):
        self.model_id = model_id
        self.file_path = file_path
        self.model_class = model_class
        self.base_url = base_url

    def __call__(self) -> nn.Module:
        file_path = self.file_path or discover_weights_file(self.base_url, self.model_id)
        dst_dir = Path("tmp") / "modelhub_downloads" / self.model_id
        ckpt_path = dst_dir / Path(file_path).name

        if not ckpt_path.exists():
            if not has_internet_connection():
                raise ConnectionError(
                    "No internet connection: cannot reach the Model Hub to pull "
                    f"model '{self.model_id}'."
                )
            ckpt_path = download_file(self.base_url, self.model_id, file_path, dst_dir)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        if self.model_class is None:
            raise ValueError(
                "model_class is required: Model Hub checkpoints store weights "
                "as a raw state_dict, not a pickled model object."
            )
        model = self.model_class()
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        return model.eval()


def has_internet_connection(timeout: float = 3.0) -> bool:
    """Checks for internet connectivity."""
    import socket

    try:
        socket.create_connection(("1.1.1.1", 443), timeout=timeout)
        return True
    except OSError:
        return False
