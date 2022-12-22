#!/bin/bash
#SBATCH --job-name=getdis
#SBATCH -N 1
##SBATCH --exclusive
##SBATCH --time=24:00:00
##SBATCH --mem=50G
#SBATCH --output=../runinfo/getdis.%j.out
#SBATCH --error=../runinfo/getdis.%j.err
##SBATCH --partition=multigpu
##module load python-2.7.5
#module load ~/.conda/envs/py37gpu/bin/python

python  test.py $1
