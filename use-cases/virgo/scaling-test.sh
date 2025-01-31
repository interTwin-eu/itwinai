#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

rm *checkpoint.pth.tar *.out *.err *.csv

timeout="03:30:00"
for N in 1 2 4 8
do
    bash runall.sh $N $timeout
    echo
done