import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import certifi
import requests
import typer

from itwinai.model_hub.feature import has_internet_connection

cli_logger = logging.getLogger("cli_logger")
py_logger = logging.getLogger(__name__)


def load_env_file(env_path: Path, env_dict: dict):
    """Load environment variables from a .env file into the provided dictionary."""
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]

                env_dict[key] = value


def upload_model_to_hub(
    model_dir: str,
    hub_url: str | None = None,
    api_token: str | None = None,
    env_file: str | None = None,
    upload_script: str | None = None,
) -> None:
    """Upload a model checkpoint to the AI Model Hub. See the `upload_model_to_hub` Typer
    command in `itwinai.cli` for the user-facing docstring.
    """
    model_path = Path(model_dir).resolve()

    # Validate if model directory exists and is a directory
    if not model_path.exists() or not model_path.is_dir():
        cli_logger.error(
            f"Model directory '{model_path}' does not exist or is not a directory."
        )
        raise typer.Exit(code=1)

    # Check if the file manifest.yaml exists
    manifest_file = model_path / "manifest.yaml"
    if not manifest_file.exists():
        cli_logger.error(f"No manifest.yaml found in '{model_path}'. ")
        raise typer.Exit(code=1)

    # Load environment variables from .env file if specified and if file exists
    env_vars = os.environ.copy()

    if env_file:
        env_path = Path(env_file)
        if not env_path.exists():
            cli_logger.error(f"Specified .env file '{env_path}' does not exist!")
            raise typer.Exit(code=1)
        load_env_file(env_path, env_vars)
        cli_logger.info(f"Loaded environment from {env_path}")
    elif Path(".env").exists():
        load_env_file(Path(".env"), env_vars)
        cli_logger.info("Loaded environment from .env file in current directory")

    # Get credentials with priority: CLI args > env vars > .env file
    final_hub_url = hub_url or env_vars.get("HYPHA_SERVER_URL")
    final_api_token = api_token or env_vars.get("HYPHA_API_TOKEN")

    if not final_hub_url:
        cli_logger.error(
            "Model hub URL not provided. Set it via:\n"
            "  - --hub-url option\n"
            "  - HYPHA_SERVER_URL environment variable\n"
            "  - HYPHA_SERVER_URL in .env file"
        )
        raise typer.Exit(code=1)

    if not final_api_token:
        cli_logger.error(
            "API token not provided. Set it via:\n"
            "  - --api-token option\n"
            "  - HYPHA_API_TOKEN environment variable\n"
            "  - HYPHA_API_TOKEN in .env file"
        )
        raise typer.Exit(code=1)

    # Update environment variables for subprocess
    env_vars["HYPHA_SERVER_URL"] = final_hub_url
    env_vars["HYPHA_API_TOKEN"] = final_api_token

    # Get or download the upload script
    upload_script_path = None
    temp_dir = None

    if upload_script:
        # User provided a path to the upload script
        upload_script_path = Path(upload_script)
        if not upload_script_path.exists():
            cli_logger.error(f"Upload script not found at: {upload_script_path}")
            raise typer.Exit(code=1)
    else:
        # Check internet connectivity
        if not has_internet_connection():
            cli_logger.warning(
                "No internet connection detected. "
                "Automatic download of upload_model.py may fail"
            )

        # Download the script from GitHub
        cli_logger.info("Downloading upload_model.py from GitHub...")
        github_url = "https://raw.githubusercontent.com/RI-SCALE/ai-model-hub-example/main/upload_model.py"

        try:
            temp_dir = tempfile.mkdtemp()
            upload_script_path = Path(temp_dir) / "upload_model.py"

            response = requests.get(github_url, timeout=15, verify=certifi.where())
            response.raise_for_status()
            upload_script_path.write_text(response.text, encoding="utf-8")

            cli_logger.info(f"Downloaded upload script to: {upload_script_path}")
        except Exception as e:
            cli_logger.error(
                "Failed to download upload script. "
                "This is likely due to missing internet access."
            )
            cli_logger.error(str(e))
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise typer.Exit(code=1)

    # Upload the model
    cli_logger.info(f"Uploading model from '{model_path}' to {final_hub_url}")

    try:
        # Build the "root/<ckpt_dir_name>" layout that discover_weights_file expects,
        # via a symlink
        scratch_dir = Path(tempfile.mkdtemp())
        root_dir = scratch_dir / "root"
        root_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = root_dir / model_path.name
        if not symlink_path.exists():
            symlink_path.symlink_to(model_path, target_is_directory=True)

        relative_upload_arg = f"root/{model_path.name}"

        try:
            # Call the upload script as subprocess
            # The original usage is: python upload_model.py model_example1
            result = subprocess.run(
                [sys.executable, str(upload_script_path), relative_upload_arg],
                env=env_vars,
                capture_output=True,
                text=True,
                cwd=str(scratch_dir),
                check=False,
            )
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

        # Print stdout (even if there's an error, this may be useful for debugging)
        if result.stdout:
            cli_logger.info(result.stdout)

        if result.returncode != 0:
            cli_logger.error(f"Upload failed with return code {result.returncode}")
            if result.stderr:
                cli_logger.error(f"Error output:\n{result.stderr}")
            raise typer.Exit(code=1)

        cli_logger.info("Model uploaded!")

    except FileNotFoundError:
        cli_logger.error(f"Python interpreter not found: {sys.executable}")
        raise typer.Exit(code=1)
    except Exception as e:
        cli_logger.error(f"Failed to upload model: {e}")
        py_logger.exception("Full error trace:")
        raise typer.Exit(code=1)
    finally:
        # Clean up temporary directory
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
