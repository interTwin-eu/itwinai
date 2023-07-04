# #!/bin/bash

IMAGE_NAME="$1"

if [ -z "$IMAGE_NAME" ]; then
<<<<<<< HEAD
    IMAGE_NAME=use-cases/mnist/containers/preproc-container.sif
fi

singularity build $IMAGE_NAME use-cases/mnist/containers/preproc-container.def
=======
    IMAGE_NAME=preproc-container.sif
fi

singularity build $IMAGE_NAME preproc-container.def
>>>>>>> 00abab4 (ported AI and preprocessing module to singularity)
