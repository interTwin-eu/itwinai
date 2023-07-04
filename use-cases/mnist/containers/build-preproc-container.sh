# #!/bin/bash

IMAGE_NAME="$1"

if [ -z "$IMAGE_NAME" ]; then
    IMAGE_NAME=preproc-container.sif
fi

singularity build $IMAGE_NAME preproc-container.def
