# #!/bin/bash

IMAGE_NAME="$1"

if [ -z "$IMAGE_NAME" ]; then
    IMAGE_NAME=base-container.sif
fi

singularity build $IMAGE_NAME base-container.def
