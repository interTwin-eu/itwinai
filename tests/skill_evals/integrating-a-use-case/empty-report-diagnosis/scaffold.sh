#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=../fixtures/lib.sh
source "$(dirname "$0")/../fixtures/lib.sh"

copy_fno_plugin

# Break the chain that feeds the GPU plots, the way it is broken in practice: GPU data is not
# measured, and the MLflow logger is left at its default of logging rank 0 only.
break_config() {
    grep -qE "$1" config.yaml || { echo "fixture config.yaml no longer matches: $1" >&2; exit 1; }
    sed -i -E "$2" config.yaml
}
break_config '^ +measure_gpu_data: true$' 's/^( +measure_gpu_data:) true$/\1 false/'
break_config '^ +log_on_workers: -1$' '/^ +log_on_workers: -1$/d'
break_config '^  mode: single$' 's/^  mode: single$/  mode: scaling-test\n  scalability_nodes: "1, 2, 4"/'
