#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="${SCRIPT_DIR}"
REPO_ROOT="$(cd "${DEMO_DIR}/../.." && pwd)"

OUTPUT_ROOT="${DEMO_DIR}/output"
TRACE_DIR="${OUTPUT_ROOT}/traces"
GOAL_DIR="${OUTPUT_ROOT}/goal"
SIM_DIR="${OUTPUT_ROOT}/sim"

TRACE_PREFIX="${TRACE_DIR}/pmpi-trace-rank-"
GOAL_FILE="${GOAL_DIR}/lulesh_8r.goal"
BIN_FILE="${GOAL_DIR}/lulesh_8r.bin"
LGS_LOG="${SIM_DIR}/loggopsim_lulesh_8r.log"
HTSIM_LOG="${SIM_DIR}/htsim_uec_lulesh_8r.log"

LIBALLPROF_SO="${REPO_ROOT}/goal_gen/hpc/liballprof/.libs/liballprof.so"
SCHEDGEN_EXEC="${REPO_ROOT}/goal_gen/hpc/Schedgen/schedgen"
TXT2BIN_EXEC="${REPO_ROOT}/sim/LogGOPSim/txt2bin"
LOGGOPSIM_EXEC="${REPO_ROOT}/sim/LogGOPSim/LogGOPSim"
HTSIM_EXEC="${REPO_ROOT}/sim/htsim-backend/sim/datacenter/htsim_uec"
LULESH_EXEC="${REPO_ROOT}/apps/hpc/lulesh/build/lulesh2.0"
HTSIM_TOPO="${REPO_ROOT}/sim/htsim-backend/sim/datacenter/topologies/leaf_spine_tiny.topo"

mkdir -p "${TRACE_DIR}" "${GOAL_DIR}" "${SIM_DIR}"

demo_require_file() {
    local path="$1"
    local hint="${2:-}"
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Required file not found: ${path}" >&2
        if [[ -n "${hint}" ]]; then
            echo "[ERROR] ${hint}" >&2
        fi
        exit 1
    fi
}

demo_info() {
    echo "[INFO] $*"
}
