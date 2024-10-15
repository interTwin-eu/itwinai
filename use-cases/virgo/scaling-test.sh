#!/bin/bash

rm *checkpoint.pth.tar *.out *.err *.csv

timeout="03:30:00"
for N in 1 2 4 8
do
    bash runall.sh $N $timeout
    echo
done