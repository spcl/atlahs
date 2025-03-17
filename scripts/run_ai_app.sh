#!/bin/bash -l
#SBATCH --job-name="run AI app proxy"
#SBATCH --nodes=1                  # number of nodes
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --partition=normal
#SBTACH --account a-g34
#SBATCH --time=12:00:00            # total run time limit (HH:MM:SS)

ATLAHS_DIR=/capstor/scratch/cscs/sshen/workspace/atlahs
SRC_DIR=${ATLAHS_DIR}/scripts
OUTPUT_FILE=Llama_N64_GPU256_PP8_DP32_70B.out
SCRIPT=${ATLAHS_DIR}/apps/ai/scripts/run_megatron.sh
TRACE_DIR=${ATLAHS_DIR}/apps/ai/scripts/nsys_reports

srun -A a-g34 --environment=megatron python3 ${SRC_DIR}/run_ai_app.py -o ${SRC_DIR}/${OUTPUT_FILE} -c "sbatch -A a-g34 ${SCRIPT} --trace atlahs" --trace-dir ${TRACE_DIR} -n 1 -v