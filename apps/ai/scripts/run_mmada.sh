#!/bin/bash -l
#SBATCH --job-name="mmada-pretrain"
#SBATCH --nodes=4                 # number of nodes
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --gpus-per-node=4          # number of gpus per node
#SBATCH --partition=normal
#SBATCH --account a-g34
#SBATCH --time=00:30:00            # total run time limit (HH:MM:SS)


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
BASE_PATH="/capstor/scratch/cscs/sshen/workspace/atlahs/apps/ai/MMaDA"
cd ${BASE_PATH}
SRC_PATH=${BASE_PATH}/training/train_mmada_stage3.py


SAVE_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}
# TOKENIZER_PATH=${BASE_PATH}/tokenizer/tokenizer.model

ACCELERATE_CONFIG_FILE=${BASE_PATH}/accelerate_configs/8_node_32_gpus_deepspeed_zero2.yaml
TRAINING_CONFIG_FILE=${BASE_PATH}/configs/mmada_pretraining_stage3_llada_instruct.yaml

# Set training command
LAUNCHER=" \
       accelerate launch \
       "

CMD="${LAUNCHER} \
         --config_file=${ACCELERATE_CONFIG_FILE} \
         --main_process_port=${MASTER_PORT} \
         --main_process_ip=${MASTER_ADDR} \
         --num_processes=$((SLURM_GPUS_PER_NODE * NNODES)) \
         --num_machines=${NNODES} \
         --machine_rank=${SLURM_NODEID} \
         --rdzv_backend=c10d \
         ${SRC_PATH} \
         config=${TRAINING_CONFIG_FILE} \
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
      #  NSYS="\
      #         nsys profile \
      #         -s none \
      #         --output="${NSYS_REPORTS_DIR}/nsys_report_%h_%p.nsys-rep" \
      #         "
       echo "ATLAHS tracing enabled, nsys reports will be saved in ${NSYS_REPORTS_DIR}"
fi


export HF_HOME=${BASE_PATH}/hf_cache
export HF_LOCAL_HOME=${HF_HOME}/local

# NSYS="\
#       nsys profile \
#       -s none \
#       --output="${NSYS_REPORTS_DIR}/nsys_report_%h_%p.nsys-rep" \
#       "


RUN="${NSYS}${CMD}"
srun --export=ALL,LD_PRELOAD=/users/sshen/workspace/nccl_goal_generator/third_party/nccl_nvtx/nccl/build/lib/libnccl.so --mpi=pmi2 --environment=ml bash -c "
export NODE_RANK=\${SLURM_NODEID}
echo ${RUN}
${RUN} 2>&1 | tee ${LOG_PATH}
"
