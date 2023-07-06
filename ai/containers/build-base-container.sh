# #!/bin/bash

IMAGE_NAME="$1"

if [ -z "$IMAGE_NAME" ]; then
    IMAGE_NAME=ai/containers/base-container.sif
fi

singularity build $IMAGE_NAME ai/containers/base-container.def
