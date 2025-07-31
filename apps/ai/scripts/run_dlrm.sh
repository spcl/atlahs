#!/bin/bash -l
#SBATCH --job-name="dlrm"
#SBATCH --nodes=4                 # number of nodes
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --gpus-per-node=1          # number of gpus per node
#SBATCH --partition=normal
#SBATCH --account a-g34
#SBATCH --time=00:05:00            # total run time limit (HH:MM:SS)


# Setting the environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1

# Extra debugging flags, slow down training
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Distributed training variables
NNODES=${SLURM_NNODES}
GPUS_PER_NODE=1
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

BASE_PATH=$(readlink -f .)
DLRM_PATH=$(readlink -f ../dlrm)
SRC_PATH=${DLRM_PATH}/dlrm_s_pytorch.py

# Set training command
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank \${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "

DLRM_ARGS=" \
       --arch-embedding-size=\"80000-80000-80000-80000-80000-80000-80000-80000\" \
       --arch-sparse-feature-size=64 \
       --arch-mlp-bot=\"128-128-128-128-64\" \
       --arch-mlp-top=\"512-512-512-256-1\" \
       --max-ind-range=40000000 \
       --data-generation=random \
       --loss-function=bce \
       --round-targets=True \
       --learning-rate=1.0 \
       --mini-batch-size=2048 \
       "

TRAINING_ARGS="\
       --print-freq=2 \
       --print-time \
       --test-freq=0 \
       --test-mini-batch-size=2048 \
       --memory-map \
       --use-gpu \
       --num-batches=18 \
       --dist-backend=nccl \
       "

PROFILING_ARGS="\
       --enable-profiling \
       --profiling-start=10 \
       --profiling-end=15 \
       "

CMD="\
       ${LAUNCHER} \
       ${SRC_PATH} \
       ${DLRM_ARGS} \
       ${TRAINING_ARGS} \
       ${PROFILING_ARGS} \
       "

# NSYS="\
#        nsys profile \
#        --trace='nvtx,cuda' \
#        --cuda-memory-usage='true' \
#        --output='${BASE_PATH}/trace_%q{SLURM_NODEID}_%q{SLURM_LOCALID}.nsys-rep' \
#        --force-overwrite true \
#        --capture-range=cudaProfilerApi \
#        --capture-range-end=stop \
#        "
NSYS=""

# NSYS="\
#        nsys profile \
#        --trace='nvtx,cuda' \
#        --output='${BASE_PATH}/trace_%q{SLURM_NODEID}_%q{SLURM_LOCALID}.nsys-rep' \
#        --force-overwrite true \
#        --capture-range=cudaProfilerApi \
#        --capture-range-end=stop \
# "

NSYS="\
       nsys profile \
       --trace='nvtx,cuda' \
       --cuda-memory-usage=false \
       --cuda-um-cpu-page-faults=false \
       --cuda-um-gpu-page-faults=false \
       -s none \
       --output='${BASE_PATH}/nsys_reports/nsys_report_%h_%p.nsys-rep' \
       "



RUN="${NSYS}${CMD}"
srun --export=ALL,LD_PRELOAD=/users/sshen/workspace/nccl_goal_generator/third_party/nccl_nvtx/nccl/build/lib/libnccl.so --mpi=pmi2 --environment=ml bash -c "
export NODE_RANK=\${SLURM_NODEID}
echo ${RUN}
${RUN} 2>&1
"
