from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import yaml
from pathlib import Path


def build_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a manifest dictionary following the Model Hub schema."""

    # Ensure required fields exist
    name = config.get("name")
    model_id = config.get("id")

    if not name or not model_id:
        raise ValueError("Model Hub config must contain at least 'name' and 'id'.")

    manifest = {
        "authors": [
            {
                "name": config.get("author_name", ""),
                "affiliation": config.get("author_affiliation", ""),
                "github_user": config.get("github_user", ""),
            }
        ],
        "covers": [],
        "description": config.get("description", ""),
        "documentation": config.get("documentation", "README.md"),
        "format_version": "0.1.0",
        "git_repo": config.get("git_repo", ""),
        "id": model_id,
        "license": config.get("license", "MIT"),
        "links": config.get("links", []),
        "maintainers": config.get("maintainers", []),
        "name": name,
        "tags": config.get("tags", []),
        "type": config.get("type", "torch"),
        "version": config.get("version", "0.0.1"),
    }

    return manifest


def write_manifest(
    ckpt_dir: Path,
    config: Dict[str, Any],
    overwrite: bool = False,
) -> Path:
    """Create manifest.yaml inside checkpoint directory."""

    manifest_path = ckpt_dir / "manifest.yaml"

    if manifest_path.exists() and not overwrite:
        return manifest_path

    manifest = build_manifest(config)

    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    return manifest_path


def write_metadata(
    ckpt_dir: Path,
    trainer_config: Any,
    overwrite: bool = False,
) -> Path:
    """Optional metadata.json for additional info."""

    import json

    metadata_path = ckpt_dir / "metadata.json"

    if metadata_path.exists() and not overwrite:
        return metadata_path

    if hasattr(trainer_config, "to_dict"):
        cfg = trainer_config.to_dict()
    else:
        cfg = str(trainer_config)

    metadata = {
        "framework": "pytorch",
        "created_at": datetime.utcnow().isoformat(),
        "training_config": cfg,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path
