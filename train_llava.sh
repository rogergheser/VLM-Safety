#!/bin/bash
#SBATCH -p edu-long
#SBATCH --job-name=LLavaTraining
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # just 1 process
#SBATCH --gres=gpu:2              # request 2 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

HF_TOKEN=$(cat hf-cli)

source ../.venv/bin/activate
huggingface-cli login --token $HF_TOKEN

echo "SLURM_JOBID: $SLURM_JOBID"
nvidia-smi

# NCCL settings not strictly needed in single-process + device_map
export CUDA_VISIBLE_DEVICES=0
export PYTHONFAULTHANDLER=1

# Just run python, no torchrun
python main.py
