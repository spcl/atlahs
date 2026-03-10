#!/bin/bash -l
#SBATCH --job-name "BUild vllm"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -A a-g34
#SBATCH --partition=normal
#SBATCH --mem=400GB
#SBATCH --time=02:00:00

export VLLM_DIR=/capstor/scratch/cscs/sshen/workspace/atlahs/apps/ai/vllm
export CMAKE_GENERATOR="Ninja"
export MAX_JOBS=16
export CMAKE_BUILD_PARALLEL_LEVEL=16
export VERBOSE=1
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDAARCHS="90"
export FLASH_ATTENTION_CUDA_ARCH_LIST="90"
export TMPDIR=/capstor/scratch/cscs/sshen/workspace/atlahs/apps/ai/vllm/.tmp
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90 -DFA2_ARCHS=9.0 -DFA3_ARCHS=9.0"
export NVCC_FLAGS="-gencode arch=compute_90,code=sm_90"
export VLLM_PREFIX=${VLLM_DIR}/install
srun --environment=vllm --export=ALL,VLLM_DIR,MAX_JOBS,CMAKE_ARGS,VERBOSE,TMPDIR,NVCC_FLAGS,VLLM_PREFIX bash -c "
cd ${VLLM_DIR} &&
pip install --prefix "${VLLM_PREFIX}" -e . --no-deps --no-build-isolation -v
"
