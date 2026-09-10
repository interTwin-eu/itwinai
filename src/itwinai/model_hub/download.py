from pathlib import Path

import requests


def list_files(base_url: str, model_id: str, subpath: str = "") -> list[dict]:
    """List files in a directory on the Model Hub."""
    url = f"{base_url}/{model_id}/files/{subpath}".rstrip("/") + "/"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def discover_weights_file(base_url: str, model_id: str) -> str:
    """Locates model.pt inside root/<ckpt_dir_name>/, following the
    save_checkpoint style in trainer.
    """
    top_level = list_files(base_url, model_id)
    if not any(e["name"] == "root" and e["type"] == "directory" for e in top_level):
        raise ValueError(
            f"Model '{model_id}' has no 'root/' directory; cannot "
            "auto-discover weights. Please specify file_path explicitly."
        )

    root_entries = list_files(base_url, model_id, "root")
    subdirs = [e["name"] for e in root_entries if e["type"] == "directory"]
    if len(subdirs) != 1:
        raise ValueError(
            f"Expected exactly one checkpoint directory under 'root/' for "
            f"model '{model_id}', found: {subdirs or '[]'}. "
            "Please specify file_path explicitly."
        )
    ckpt_dir_name = subdirs[0]

    ckpt_entries = list_files(base_url, model_id, f"root/{ckpt_dir_name}")
    filenames = [e["name"] for e in ckpt_entries if e["type"] == "file"]
    if "model.pt" not in filenames:
        raise ValueError(
            f"No 'model.pt' found under 'root/{ckpt_dir_name}/' for model "
            f"'{model_id}'. Found: {filenames}. "
            "Please specify file_path explicitly."
        )
    return f"root/{ckpt_dir_name}/model.pt"


def download_file(base_url: str, model_id: str, file_path: str, dst_dir: Path) -> Path:
    """Download a file from the Model Hub."""
    url = f"{base_url}/{model_id}/files/{file_path}"
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise ValueError(
            f"Could not download '{file_path}' for model '{model_id}' "
            f"from the Model Hub (status {response.status_code})."
        )
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / Path(file_path).name
    dst_path.write_bytes(response.content)
    return dst_path
