#!/bin/bash
set -e

# The command to launch, defaulting to the image's CMD.
notebook_cmd=("$@")
if [[ ${#notebook_cmd[@]} -eq 0 ]]; then
    notebook_cmd=(start-notebook.sh)
fi

echo "[start.sh] Running setup.sh for Rucio (generates rucio.cfg)..."
if ! /usr/local/bin/setup.sh; then
    echo "[start.sh] WARNING: Rucio setup failed. Starting anyway: the Rucio client and its" \
        "JupyterLab extension will be unconfigured, everything else works." >&2
fi

# Kubeflow serves each notebook behind a path prefix and passes it in NB_PREFIX. The upstream
# Jupyter entrypoint ignores that variable, so without these flags the server starts and every
# request returns 404. NOTEBOOK_ARGS is the hook that start-notebook.py reads.
if [[ -n "${NB_PREFIX:-}" ]]; then
    echo "[start.sh] Kubeflow detected: serving JupyterLab under ${NB_PREFIX}"
    kf_args="--ServerApp.base_url=${NB_PREFIX}"
    kf_args="${kf_args} --ServerApp.token= --ServerApp.password="
    kf_args="${kf_args} --ServerApp.allow_origin=* --ServerApp.ip=0.0.0.0"
    export NOTEBOOK_ARGS="${kf_args} ${NOTEBOOK_ARGS:-}"
fi

# Platforms that run the container as the logged-in user, Kubeflow among them, replace
# /etc/passwd with one that has no jovyan user. The upstream start.sh looks up that user under
# `set -e` and dies with no message. Skip it: it only switches user and runs hook directories,
# and here we already run as the right user.
if ! id -u jovyan > /dev/null 2>&1; then
    echo "[start.sh] No jovyan user in /etc/passwd: launching the notebook directly."
    exec "${notebook_cmd[@]}"
fi

echo "[start.sh] Running original start.sh..."
exec /usr/local/bin/start-original.sh "${notebook_cmd[@]}"
