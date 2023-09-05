# #!/bin/bash

IMAGE_NAME="$1"

if [ -z "$IMAGE_NAME" ]; then
    IMAGE_NAME=train-container.sif
fi

singularity build $IMAGE_NAME train-container.def
