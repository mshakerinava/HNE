#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

module load httpproxy
source $HOME/venv_py38/bin/activate
python main.py $@

