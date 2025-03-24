#!/bin/bash -l
#SBATCH --job-name="nccl goal gen"
#SBATCH --nodes=1                  # number of nodes
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --partition=debug
#SBTACH --account a-g34

ATLAHS_DIR=/capstor/scratch/cscs/sshen/workspace/atlahs
SRC_DIR=${ATLAHS_DIR}/goal_gen/ai/nccl_goal_generator
SCRIPT=${SRC_DIR}/get_traced_events.py
INPUT_DIR=${ATLAHS_DIR}/data/MoE_N64_GPU256_TP4_PP8_DP8_EP8_70B_allgather/trial_5/nsys_reports
# INPUT_DIR=${ATLAHS_DIR}/data/MoE_N32_GPU128_TP4_PP4_DP8_EP4_13B/trial_1/nsys_reports
DATA_DIR=/capstor/scratch/cscs/sshen/workspace/goal_data
OUTPUT_DIR=${DATA_DIR}/validation/ai/moe/N64_GPU256_2
# OUTPUT_DIR=${DATA_DIR}/validation/ai/moe/N32_GPU128_alltoall
NPKIT_SIMPLE=${ATLAHS_DIR}/goal_gen/ai/nccl_goal_generator/npkit_benchmark_results/clariden/npkit_data_summary_Simple.json
NPKIT_LL=${ATLAHS_DIR}/goal_gen/ai/nccl_goal_generator/npkit_benchmark_results/clariden/npkit_data_summary_LL.json

srun -A a-g34 --environment=megatron python3 ${SCRIPT} -i ${INPUT_DIR} -o ${OUTPUT_DIR} -q --unique-nic -s ${NPKIT_SIMPLE} -l ${NPKIT_LL}