#!/bin/bash

docker run --rm -it \
    -v $PWD:/data \
    -u $(id -u):$(id -g) \
    openjournals/inara \
    -o pdf,crossref \
    paper.md