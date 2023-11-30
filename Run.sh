#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/log.out
#SBATCH --job-name=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100

python main-copy.py
