#!/bin/bash
#SBATCH -p edu-long
#SBATCH --job-name=LLavaTraining
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                # Number of processes = number of GPUs
#SBATCH --gres=gpu:0              # Request 2 GPUs
#SBATCH --cpus-per-task=8         # Number of CPU cores per process
#SBATCH --mem=0                   # Use full node memory
HF_TOKEN=$(cat hf-cli)

source .venv/bin/activate
huggingface-cli login --token $HF_TOKEN

srun python data_module.py
