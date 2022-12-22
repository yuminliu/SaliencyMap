#!/bin/bash
for gcm in {0..3}
do
    work=/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/src
    cd $work
    sbatch runpy.bash $gcm
done
