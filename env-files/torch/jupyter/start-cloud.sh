#!/bin/bash
set -e

echo "[start.sh] Running setup.sh for Rucio (generates rucio.cfg)..."
/usr/local/bin/setup.sh

echo "[start.sh] Running original start.sh..."
exec /usr/local/bin/start-original.sh "$@"
