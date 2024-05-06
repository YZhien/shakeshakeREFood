#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=16G
#SBATCH --time=20:00:00


module purge
module load gcc/11.3.0
module load python/3.9.12
module load nvhpc/22.11

cd ./567_shake_RE
eval "$(conda shell.bash hook)"
conda activate 567project

python train.py --depth 26 --base_channels 32 --shake_forward True --shake_backward True --shake_image True --outdir results


