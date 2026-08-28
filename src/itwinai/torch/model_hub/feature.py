from pathlib import Path

from itwinai.torch.model_hub.backends import get_backend
from itwinai.torch.model_hub.manifest import write_manifest
from itwinai.torch.model_hub.utils import has_internet_connection


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
        if ckpt_dir.name != self.final_checkpoint_name:
            return
        write_manifest(ckpt_dir, self.config)

    def on_training_end(self, trainer, ckpt_dir):
        if not self.enabled:
            return
        ckpt_dir = Path(ckpt_dir)
        if ckpt_dir.name != self.final_checkpoint_name:
            return
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
