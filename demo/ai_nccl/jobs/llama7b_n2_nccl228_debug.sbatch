#!/bin/bash -l
#SBATCH --job-name=demo-llama7b-n2-228
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=debug
#SBATCH --account=a-g200
#SBATCH --time=00:20:00
#SBATCH --output=/iopsstor/scratch/cscs/btommaso/DemoAI/demo/ai_nccl/cache/trace_job/slurm/slurm_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/btommaso/DemoAI/demo/ai_nccl/cache/trace_job/slurm/slurm_%j.err

set -euo pipefail

TRACE_DIR=/iopsstor/scratch/cscs/btommaso/DemoAI/demo/ai_nccl/cache/trace_job/raw_nsys
BASE_PATH=/iopsstor/scratch/cscs/btommaso/DemoAI/demo/ai_nccl/cache/trace_job/workspace
mkdir -p "${TRACE_DIR}" "${BASE_PATH}"
rm -f "${TRACE_DIR}"/*.nsys-rep

export TP=1
export PP=1
export MODEL_SIZE=7
export MICRO_BATCH_SIZE=4
export TRACE_DIR
export BASE_PATH

bash /iopsstor/scratch/cscs/btommaso/DemoAI/apps/ai/scripts/run_megatron.sh --trace atlahs --nccl 228
