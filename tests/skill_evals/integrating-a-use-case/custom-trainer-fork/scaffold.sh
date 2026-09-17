#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=../fixtures/lib.sh
source "$(dirname "$0")/../fixtures/lib.sh"

copy_plugin_template
cp "$REPO_ROOT/tutorials/claude-skill/fno-darcy/train.py" train.py
