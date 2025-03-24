#!/bin/bash -l
#SBATCH --job-name="Chakra trace generation"
#SBATCH --nodes=1                  # number of nodes
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --partition=debug
#SBTACH --account a-g34

ATLAHS_DIR=/capstor/scratch/cscs/sshen/workspace/atlahs
SRC_DIR=${ATLAHS_DIR}/scripts
SCRIPT=${SRC_DIR}/et_to_chakra.sh
INPUT_DIR=${ATLAHS_DIR}/data/astrasim_traces/MoE_N64_GPU256_TP4_PP8_DP8_EP8_70B_allgather
OUTPUT_DIR=${ATLAHS_DIR}/data/chakra_traces/MoE_N64_GPU256

srun -A a-g34 --environment=megatron bash ${SCRIPT} -i ${INPUT_DIR} -o ${OUTPUT_DIR} -r --max-conversion-threads 16