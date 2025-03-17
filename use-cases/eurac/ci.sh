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
bash ray_cluster_container.sh
singularity exec  -B /ceph/hpc/data/st2301-itwin-users/eurac/config.yaml:/app/eurac/config.yaml \
    eurac.sif /bin/bash -c \
    'cd /app/eurac && python hpo.py --workdir /ceph/hpc/data/st2301-itwin-users/mbunino \
    --dataset_path /ceph/hpc/data/st2301-itwin-users/eurac \
    --ncpus 8 \
    --ngpus 1 \
    --max_iterations 20 \
    --pipeline_name training_pipeline'