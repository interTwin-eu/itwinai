#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

CMD="itwinai exec-pipeline"

# Run command in the itwinai torch Docker container
if [ -z "$1" ]; then
    # CPU only execution
    docker run -it --rm --name mnist-training --user $UID:$GID \
        --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "$PWD":/use-case  ghcr.io/intertwin-eu/itwinai:0.2.2-torch-2.1 \
        /bin/bash -c "cd /use-case && $CMD"
elif [ "$1" == "gpu" ]; then
    # With GPU support: --gpus all
    docker run -it --rm --name mnist-training --user $UID:$GID \
        --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "$PWD":/use-case  ghcr.io/intertwin-eu/itwinai:0.2.2-torch-2.1 \
        /bin/bash -c "cd /use-case && $CMD"
fi
