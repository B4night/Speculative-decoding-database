#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=350GB
#SBATCH --gres=gpu:a100:4

python t.py