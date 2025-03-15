#!/bin/bash -l
#SBATCH --job-name="megatron-llama2-pretrain"
#SBATCH --nodes=4                 # number of nodes
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --gpus-per-node=4          # number of gpus per node
#SBATCH --partition=normal
#SBTACH --account a-g34
#SBATCH --time=00:10:00            # total run time limit (HH:MM:SS)


# Default trace value (used only if --trace is provided)
trace="atlahs"
# By default, tracing is disabled
USE_TRACING=false

usage() {
  echo "Usage: $0 [--trace atlahs|astrasim]"
  exit 1
}

# Process command-line arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --trace)
      # Ensure a value follows the flag
      if [[ -n "$2" && "$2" != --* ]]; then
        trace="$2"
        USE_TRACING=true
        shift 2
      else
        echo "Error: --trace requires a value."
        usage
      fi
      ;;
    --trace=*)
      trace="${1#*=}"
      USE_TRACING=true
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      usage
      ;;
  esac
done

# If tracing is enabled, validate the trace value
if $USE_TRACING; then
  if [[ "$trace" != "atlahs" && "$trace" != "astrasim" ]]; then
    echo "Error: --trace must be either 'atlahs' or 'astrasim'."
    usage
  fi
  echo "Tracing enabled with value: $trace"
else
  echo "Tracing disabled."
fi


# Setting the environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1

# Extra debugging flags, slow down training
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Distributed training variables
NNODES=${SLURM_NNODES}
GPUS_PER_NODE=4
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Parallelism variables
TP=1
PP=1
DP=$((${GPU_NUM}/${TP}/${PP}))
MICRO_BATCH_SIZE=1
# Global batch size should be the maximum of MICRO_BATCH_SIZE * DP or 32
GLOBAL_BATCH_SIZE=$(((${MICRO_BATCH_SIZE}*${DP}) > 32 ? ${MICRO_BATCH_SIZE}*${DP} : 32))
echo "Global batch size: ${GLOBAL_BATCH_SIZE}"
echo "Micro batch size: ${MICRO_BATCH_SIZE}"
# Network size variables
MODEL_SIZE=7

if   [[ ${MODEL_SIZE} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=4; NUM_QUERY_GROUP=4; NUM_LAYERS=4; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

DROP_OUT=0.0
MAX_LR=3e-5
MIN_LR=3e-6
MAX_SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=${MAX_SEQ_LEN}

# Paths
BASE_PATH="/users/sshen/workspace/megatron"
source ${BASE_PATH}/source_megatron.sh
cd ${BASE_PATH}
SRC_PATH=${MEGATRON_PATH}/pretrain_gpt.py

LOG_NAME=llama2-${MODEL_SIZE}b_pretrain_WS${WORLD_SIZE}_TP${TP}_PP${PP}
LOG_DIR=${BASE_PATH}/log/${LOG_NAME}
LOG_PATH=${LOG_DIR}/node${NODE_RANK}.log
mkdir -p ${LOG_DIR}

# DATA_PATH=${BASE_PATH}/data/oscar-en-10k-meg-llama_text_document
DATA_CACHE_PATH="${BASE_PATH}/data_cache/${LOG_NAME}"
mkdir -p ${DATA_CACHE_PATH}

SAVE_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}
# TOKENIZER_PATH=${BASE_PATH}/tokenizer/tokenizer.model

# Set training command
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank \${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "

DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --distributed-backend nccl \
       --use-distributed-optimizer \
       --sequence-parallel \
       --overlap-param-gather \
       --overlap-grad-reduce \
       "    

NETWORK_SIZE_ARGS=" \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEAD} \
       --group-query-attention \
       --num-query-groups ${NUM_QUERY_GROUP} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --position-embedding-type rope \
       --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
       --make-vocab-size-divisible-by 64 \
       --norm-epsilon ${NORM_EPS} \
       --normalization RMSNorm \
       --swiglu \
       --untie-embeddings-and-output-weights \
       "

LOGGING_ARGS=" \
       --log-throughput \
       --timing-log-level 0 \
       --log-timers-to-tensorboard \
       --log-validation-ppl-to-tensorboard \
       --log-memory-to-tensorboard \
       --log-world-size-to-tensorboard \
       "

REGULATIZATION_ARGS=" \
       --attention-dropout ${DROP_OUT} \
       --hidden-dropout ${DROP_OUT} \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1e-8 \
       "

TRAINING_ARGS=" \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters 12 \
    --log-interval 1 \
    --disable-bias-linear \
    --cross-entropy-loss-fusion \
    --use-flash-attn \
    --optimizer adam \
    --exit-interval 100 \
    --no-check-for-nan-in-loss-and-grad \
    --tensorboard-dir ${BASE_PATH} \
    "

if [[ "$USE_TRACING" == true && "$trace" == "atlahs" ]]; then
       TRAINING_ARGS=" \
              ${TRAINING_ARGS} \
              --profile \
              --profile-step-start 8 \
              --profile-step-end 10 \
              "
fi

if [[ "$USE_TRACING" == true && "$trace" == "astrasim" ]]; then
       TRAINING_ARGS=" \
              ${TRAINING_ARGS} \
              --use-pytorch-profile \
              --profile-step-start 8 \
              --profile-step-end 10 \
              "
fi

#     --use-pytorch-profiler \
#     --profile \
#     --profile-step-start 3 \
#     --profile-step-end 4 \
#     --profile-ranks 0 4 8 12 \

INITIALIZATION_ARGS=" \
       --seed 1403 \
       --init-method-std 0.02 \
       "

LEARNING_RATE_ARGS=" \
       --lr ${MAX_LR} \
       --lr-decay-style cosine \
       --lr-warmup-fraction 0.1 \
       --min-lr ${MIN_LR} \
       "

CHECKPOINTING_ARGS=" \
       --finetune \
       --no-load-optim \
       --no-load-rng \
       "

MIXED_PRECISION_ARGS=" \
       --bf16 \
       "

VALIDATION_ARGS=" \
       --eval-interval 1000 \
       --eval-iters 0 \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 949,50,1 \
       --seq-length ${MAX_SEQ_LEN} \
       --num-workers 0 \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model ${TOKENIZER_PATH} \
       --data-cache-path ${DATA_CACHE_PATH} \
       "

TE_ARGS=" \
    --transformer-impl transformer_engine \
    "

CMD="\
       ${LAUNCHER} \
       ${SRC_PATH} \
       ${DISTRIBUTED_ARGS} \
       ${NETWORK_SIZE_ARGS} \
       ${LOGGING_ARGS} \
       ${REGULATIZATION_ARGS} \
       ${TRAINING_ARGS} \
       ${INITIALIZATION_ARGS} \
       ${LEARNING_RATE_ARGS} \
       ${CHECKPOINTING_ARGS} \
       ${MIXED_PRECISION_ARGS} \
       ${VALIDATION_ARGS} \
       ${DATA_ARGS} \
       ${MOE_ARGS} \
       ${TE_ARGS} \
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

if [[ "$USE_TRACING" == true && "$trace" == "atlahs" ]]; then

       NSYS_REPORTS_DIR=/capstor/scratch/cscs/sshen/workspace/atlahs/apps/ai/scripts
       # Checks if the directory exists, if it doesn't, make it
       if [ ! -d "${NSYS_REPORTS_DIR}" ]; then
              mkdir -p ${NSYS_REPORTS_DIR}
       fi
       NSYS="\
              nsys profile \
              --trace='nvtx,cuda' \
              --cuda-memory-usage=false \
              --force-overwrite true \
              --cuda-um-cpu-page-faults=false \
              --cuda-um-gpu-page-faults=false \
              -s none \
              --output='${NSYS_REPORTS_DIR}/nsys_reports/nsys_report_%h_%p.nsys-rep' \
              "
       echo "ATLAHS tracing enabled, nsys reports will be saved in ${NSYS_REPORTS_DIR}"
fi



RUN="${NSYS}${CMD}"
srun --export=ALL,LD_PRELOAD=/users/sshen/workspace/nccl_goal_generator/third_party/nccl_nvtx/nccl/build/lib/libnccl.so --mpi=pmi2 --environment=megatron bash -c "
export NODE_RANK=\${SLURM_NODEID}
echo ${RUN}
${RUN} 2>&1 | tee ${LOG_PATH}
"
