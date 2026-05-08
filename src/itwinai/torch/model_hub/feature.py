from pathlib import Path

from itwinai.torch.model_hub.manifest import write_manifest, write_metadata
from itwinai.torch.model_hub.backends import get_backend


class ModelHubFeature:
    def __init__(self, config: dict):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)

        backend_name = self.config.get("backend", "itwinai")
        self.backend = get_backend(backend_name, self.config)

    def on_checkpoint_saved(self, trainer, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)

        # prepare directory
        write_manifest(ckpt_dir, self.config)
        write_metadata(ckpt_dir, trainer.config)

        # upload type
        mode = self.config.get("mode", "deferred")

        if mode == "online":
            self.backend.upload(ckpt_dir)

        elif mode == "auto":
            if self._has_internet():
                self.backend.upload(ckpt_dir)
            else:
                print(f"Model Hub config ready in: {ckpt_dir}")

        elif mode == "deferred":
            print(f"Model Hub can be run in: {ckpt_dir}")

    def _has_internet(self, timeout=3.0):
        import socket

        try:
            socket.create_connection(("8.8.8.8", 53), timeout=timeout)
            return True
        except OSError:
            return False
