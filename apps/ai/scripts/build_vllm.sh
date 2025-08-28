#!/bin/bash -l
#SBATCH --job-name "BUild vllm"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -A a-g34
#SBATCH --partition=normal
#SBATCH --mem=400GB
#SBATCH --time=02:00:00

export VLLM_DIR=/capstor/scratch/cscs/sshen/workspace/spcl-atlahs/apps/ai/vllm
export MAX_JOBS=24
export VERBOSE=1
export TORCH_CUDA_ARCH_LIST=90
export CUDAARCHS=90
export FLASH_ATTENTION_CUDA_ARCH_LIST=90
export TMPDIR=/capstor/scratch/cscs/sshen/workspace/spcl-atlahs/apps/ai/vllm/.tmp
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90 -DFA2_ARCHS=90 -DFA3_ARCHS=90"

srun --environment=vllm2 --export=ALL,VLLM_DIR,MAX_JOBS,CMAKE_ARGS,VERBOSE,TMPDIR bash -c "
cd ${VLLM_DIR} &&
pip install -e . --no-deps --no-build-isolation -v
"