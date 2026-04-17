#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CACHE_ROOT="${SCRIPT_DIR}/cache"
RUN_ROOT="${SCRIPT_DIR}/runs"
mkdir -p "${CACHE_ROOT}" "${RUN_ROOT}"

CASE="llama7b_n2"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NCCL_V2_WORKERS="${NCCL_V2_WORKERS:-$(nproc)}"
NSYS_BIN="${NSYS_BIN:-/iopsstor/scratch/cscs/btommaso/llamp_eval_20260320/tools/nsight/extracted_arm64/opt/nvidia/nsight-systems/2025.4.1/target-linux-sbsa-armv8/nsys}"

CASE_LABEL="Llama7B_N2_GPU8_TP1_PP1_DP8_BS32"
REF_CASE_ROOT="/iopsstor/scratch/cscs/btommaso/llamp_eval_20260320/cases/llama7b_n2"
GPU_RANKS=8
SLURM_NODES=2
HARDWARE_RUNTIME_NS=3343426880
REFERENCE_LGS_NS=3474093263
HTSIM_NODES=8
HTSIM_TOPO_DEFAULT="${SCRIPT_DIR}/topologies/leaf_spine_8_1os.topo"
JOB_SCRIPT="${SCRIPT_DIR}/jobs/llama7b_n2_nccl228_debug.sbatch"

TRACE_CACHE_ROOT="${CACHE_ROOT}/trace_job"
REFERENCE_LINK="${CACHE_ROOT}/reference"
RAW_NSYS_DIR="${TRACE_CACHE_ROOT}/raw_nsys"
TRACE_WORKSPACE_DIR="${TRACE_CACHE_ROOT}/workspace"
TRACE_SLURM_DIR="${TRACE_CACHE_ROOT}/slurm"

RUN_CASE_ROOT="${RUN_ROOT}/${CASE}"
SQLITE_DIR="${RUN_CASE_ROOT}/sqlite"
GOAL_DIR="${RUN_CASE_ROOT}/goal_v2"
SIM_DIR="${RUN_CASE_ROOT}/sim"

GOAL_FILE="${GOAL_DIR}/output.goal"
BIN_FILE="${SIM_DIR}/${CASE}.bin"
LGS_LOG="${SIM_DIR}/loggopsim.log"
HTSIM_LOG="${SIM_DIR}/htsim_uec.log"
HTSIM_SUMMARY_JSON="${SIM_DIR}/htsim_summary.json"

NSYS_TO_SQLITE_SCRIPT="${REPO_ROOT}/scripts/nsys_reports_to_sqlite.sh"
NCCL_V2_ROOT="${REPO_ROOT}/goal_gen/ai/nccl_generator_v2"
NCCL_V2_MAIN="${NCCL_V2_ROOT}/main.py"
NPKIT_SIMPLE="${NCCL_V2_ROOT}/npkit_benchmark_results/clariden/npkit_data_summary_Simple.json"
NPKIT_LL="${NCCL_V2_ROOT}/npkit_benchmark_results/clariden/npkit_data_summary_LL.json"
TXT2BIN_EXEC="${REPO_ROOT}/sim/LogGOPSim/txt2bin"
LOGGOPSIM_EXEC="${REPO_ROOT}/sim/LogGOPSim/LogGOPSim"
HTSIM_EXEC="${REPO_ROOT}/sim/htsim-backend/sim/datacenter/htsim_uec"
HTSIM_TOPO="${HTSIM_TOPO:-${HTSIM_TOPO_DEFAULT}}"

LGS_L="${LGS_L:-3700}"
LGS_o="${LGS_o:-200}"
LGS_g="${LGS_g:-5}"
LGS_O="${LGS_O:-0}"
LGS_G="${LGS_G:-0.04}"
LGS_S="${LGS_S:-0}"

HTSIM_LINKSPEED="${HTSIM_LINKSPEED:-200000}"
HTSIM_MTU="${HTSIM_MTU:-4096}"
HTSIM_PATHS="${HTSIM_PATHS:-128}"
HTSIM_QUEUE="${HTSIM_QUEUE:-1000000}"
HTSIM_SEED="${HTSIM_SEED:-4}"

export NSYS_BIN

demo_info() {
    echo "[INFO] $*"
}

demo_require_file() {
    local path="$1"
    local hint="${2:-}"
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing file: ${path}" >&2
        if [[ -n "${hint}" ]]; then
            echo "[ERROR] ${hint}" >&2
        fi
        exit 1
    fi
}

demo_require_dir() {
    local path="$1"
    local hint="${2:-}"
    if [[ ! -d "${path}" ]]; then
        echo "[ERROR] Missing directory: ${path}" >&2
        if [[ -n "${hint}" ]]; then
            echo "[ERROR] ${hint}" >&2
        fi
        exit 1
    fi
}

demo_prepare_layout() {
    mkdir -p \
        "${TRACE_CACHE_ROOT}" \
        "${RAW_NSYS_DIR}" \
        "${TRACE_WORKSPACE_DIR}" \
        "${TRACE_SLURM_DIR}" \
        "${RUN_CASE_ROOT}" \
        "${SQLITE_DIR}" \
        "${GOAL_DIR}" \
        "${SIM_DIR}"

    if [[ ! -e "${REFERENCE_LINK}" ]]; then
        ln -s "${REF_CASE_ROOT}" "${REFERENCE_LINK}"
    fi

    if ! compgen -G "${RAW_NSYS_DIR}/*.nsys-rep" > /dev/null; then
        local rep
        for rep in "${REF_CASE_ROOT}"/raw_nsys/*.nsys-rep; do
            ln -sf "${rep}" "${RAW_NSYS_DIR}/$(basename "${rep}")"
        done
    fi
}
