#!/bin/bash -l
#SBATCH --job-name="vllm"
#SBATCH --nodes=2                 # number of nodes
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --gpus-per-node=4          # number of gpus per node
#SBATCH --partition=normal
#SBATCH --account a-g34
#SBATCH --time=00:20:00            # total run time limit (HH:MM:SS)
export LLM_MODEL_DIR=/capstor/scratch/cscs/sshen/workspace/spcl-atlahs/apps/ai/llm-models

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=120
export OMP_NUM_THREADS=1

VLLM_DIR=/capstor/scratch/cscs/sshen/workspace/spcl-atlahs/apps/ai/vllm
INFERENCE_SCRIPT=/capstor/scratch/cscs/sshen/workspace/spcl-atlahs/apps/ai/scripts/vllm_inference.py

NNODES=${SLURM_NNODES}
GPUS_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))


# If the number of nodes is more than 1, use torch.distributed.run
# If the number of nodes is 1, use python3
if [ ${NNODES} -gt 1 ]; then
       LAUNCHER=" \
              torchrun \
                     --nproc_per_node ${GPUS_PER_NODE} \
                     --nnodes ${NNODES} \
                     --node_rank \${NODE_RANK} \
                     --master_addr ${MASTER_ADDR} \
                     --master_port ${MASTER_PORT} \
                     --rdzv_backend=c10d \
                     --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
       "
       DISTRIBUTED_EXECUTOR_BACKEND="external_launcher"
else
       LAUNCHER="python3"
       DISTRIBUTED_EXECUTOR_BACKEND="mp"
fi


PPL=1
# TP should be equal to the total number of GPUs / GPUS_PER_NODE
TP=$(expr ${NNODES} \* ${GPUS_PER_NODE})

SCRIPT_ARGS="\
       --distributed-executor-backend ${DISTRIBUTED_EXECUTOR_BACKEND} \
       --tensor-parallel-size ${TP} \
       --max-model-len 4096 \
       --gpu-memory-utilization 0.8 \
       "

INFERENCE_CMD="\
       ${LAUNCHER} \
       ${INFERENCE_SCRIPT} \
       ${SCRIPT_ARGS} \
       "

TRACE_DIR=/capstor/scratch/cscs/sshen/workspace/spcl-atlahs/apps/ai/scripts/traces

if [ ! -d "${TRACE_DIR}" ]; then
       mkdir -p ${TRACE_DIR}
fi

NSYS="\
       nsys profile \
       --trace='nvtx,cuda' \
       --trace-fork-before-exec=true \
       --cuda-memory-usage=false \
       --force-overwrite true \
       --cuda-um-cpu-page-faults=false \
       --cuda-um-gpu-page-faults=false \
       -s none \
       --output='${TRACE_DIR}/nsys_report_%h_%p.nsys-rep' \
       "

# NSYS=""

# INSTALL_VLLM_CMD="cd ${VLLM_DIR} && pip install -e . && cd - && "
INSTALL_VLLM_CMD=""

HF_TOKEN="ADD YOUR HF TOKEN HERE"

RUN="${INSTALL_VLLM_CMD}${NSYS}${INFERENCE_CMD}"

export VLLM_FLASH_ATTN_VERSION=2

# nsys profile --trace='cuda,nvtx' --cuda-memory-usage=false --cuda-um-cpu-page-faults=false --cuda-um-gpu-page-faults=false -s none --output='traces/nsys_reports_%h_%p.nsys-rep' python3 vllm_inference.py
# export LD_PRELOAD=/users/sshen/workspace/nccl_goal_generator/third_party/nccl_nvtx/nccl/build/lib/libnccl.so
srun --export=ALL,LLM_MODEL_DIR,VLLM_FLASH_ATTN_VERSION --mpi=pmi2 --environment=vllm2 bash -c "
export PYTHONPATH=${VLLM_DIR}:$PYTHONPATH
export NODE_RANK=\${SLURM_NODEID}
export HF_TOKEN=${HF_TOKEN}
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_CACHE_ROOT=${VLLM_DIR}/vllm-cache
export LD_PRELOAD=/users/sshen/workspace/nccl_goal_generator/third_party/nccl_nvtx/nccl/build/lib/libnccl.so
echo ${RUN}
${RUN} 2>&1
"