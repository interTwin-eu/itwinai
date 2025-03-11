#!/bin/bash

# shellcheck disable=all

# Run this from the repo's root

cd ci

dagger call \
    --tag eurac-hpo \
    build-container --context=.. --dockerfile=../use-cases/eurac/Dockerfile \
    publish \
    logs
    # container \
    # terminal 


exit 0

# commands on vega

itwin
singularity pull --force eurac.sif docker://ghcr.io/intertwin-eu/itwinai-dev:eurac-hpo
# Lauch ray cluster on login
bash ray_cluster_container.sh
# Launch HPO
singularity exec eurac.sif /bin/bash -c \
    'cd /app/eurac && python hpo.py --workdir /ceph/hpc/data/st2301-itwin-users/mbunino --dataset_path /ceph/hpc/data/st2301-itwin-users/eurac --ncpus 1 --ngpus 1 --max_iterations 40'