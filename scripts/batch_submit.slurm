#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=a100
