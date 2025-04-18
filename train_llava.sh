#!/bin/bash
#SBATCH -p edu-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1
#SBATCH --job-name=LLavaTraining
#SBATCH -N 1
HF_TOKEN=$(cat hf-cli)

source .venv/bin/activate
huggingface-cli login --token $HF_TOKEN
python main.py