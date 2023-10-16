# #!/bin/bash

IMAGE_NAME="$1"

if [ -z "$IMAGE_NAME" ]; then
<<<<<<< HEAD
    IMAGE_NAME=ai/containers/base-container.sif
fi

singularity build $IMAGE_NAME ai/containers/base-container.def
=======
    IMAGE_NAME=base-container.sif
fi

singularity build $IMAGE_NAME base-container.def
>>>>>>> 00abab4 (ported AI and preprocessing module to singularity)
