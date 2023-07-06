# #!/bin/bash

IMAGE_NAME="$1"

if [ -z "$IMAGE_NAME" ]; then
    IMAGE_NAME=use-cases/mnist/containers/preproc-container.sif
fi

singularity build $IMAGE_NAME use-cases/mnist/containers/preproc-container.def
