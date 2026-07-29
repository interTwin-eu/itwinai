from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_MANIFEST = {
    "format_version": "0.1.0",
    "published": "true",
    "documentation": "README.md",
    "license": "MIT",
    "git_repo": "",
    "covers": [],
    "links": [],
    "maintainers": [],
    "tags": [],
    "authors": [],
    "type": "torch",
    "version": "0.0.1",
}


def write_manifest(
    ckpt_dir: Path,
    model_hub_config: dict[str, Any],
    overwrite: bool = False,
) -> Path:
    """Create a Model Hub manifest.yaml inside the checkpoint directory."""

    manifest_path = ckpt_dir / "manifest.yaml"

    if manifest_path.exists() and not overwrite:
        return manifest_path

    # Set default values
    manifest = DEFAULT_MANIFEST.copy()

    # Include user-defined fields
    user_manifest = model_hub_config.get("manifest", {})
    manifest.update(user_manifest)

    # Validate required fields
    required_fields = ("id", "name")

    missing = [field for field in required_fields if not manifest.get(field)]

    if missing:
        raise ValueError("Missing required Model Hub manifest fields: " + ", ".join(missing))

    with open(manifest_path, "w") as f:
        yaml.safe_dump(
            manifest,
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    return manifest_path
