#!/bin/bash
#SBATCH -p amd_512
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G

module load miniforge
module load intel/2022.1
source activate pyscf-2.5

python -u $1
