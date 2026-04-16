#!/usr/bin/env bash

set -euo pipefail

# Stage 1 of the demo:
# 1. Run LULESH with 8 MPI ranks.
# 2. Preload liballprof.so so MPI calls are traced through PMPI.
# 3. Force liballprof to write all trace files into demo/output/traces.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

demo_require_file "${LIBALLPROF_SO}" "Build liballprof first, for example from goal_gen/hpc/liballprof/."
demo_require_file "${LULESH_EXEC}" "Build LULESH first, for example in apps/hpc/lulesh/build/."

MPI_RANKS="${MPI_RANKS:-8}"
LULESH_ITERS="${LULESH_ITERS:-100}"
LULESH_SIZE="${LULESH_SIZE:-16}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ "${MPI_RANKS}" != "8" ]]; then
    echo "[ERROR] This demo script is intentionally set up for 8 MPI ranks." >&2
    exit 1
fi

# Remove any old traces from a previous demo run so we can verify we created
# exactly 8 fresh files this time.
rm -f "${TRACE_DIR}"/pmpi-trace-rank-*.txt

export LD_PRELOAD="${LIBALLPROF_SO}"
# liballprof appends "<rank>.txt" to this prefix, so the trailing dash matters.
export HTOR_PMPI_FILE_PREFIX="${TRACE_PREFIX}"
export OMP_NUM_THREADS

mpirun_cmd=(mpirun)

# Open MPI usually requires an explicit opt-in when running as root in a
# container. Keep the script portable by enabling it only when needed.
if [[ "$(id -u)" -eq 0 ]]; then
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
    mpirun_cmd+=(--allow-run-as-root)
fi

mpirun_cmd+=(
    --oversubscribe
    -n "${MPI_RANKS}"
    -x LD_PRELOAD
    -x HTOR_PMPI_FILE_PREFIX
    -x OMP_NUM_THREADS
    "${LULESH_EXEC}"
    -i "${LULESH_ITERS}"
    -s "${LULESH_SIZE}"
)

demo_info "Tracing LULESH with ${MPI_RANKS} MPI ranks"
demo_info "Trace files will be written under ${TRACE_DIR}"
demo_info "Running: ${mpirun_cmd[*]}"
"${mpirun_cmd[@]}"

trace_count="$(find "${TRACE_DIR}" -maxdepth 1 -type f -name 'pmpi-trace-rank-*.txt' | wc -l | tr -d ' ')"
if [[ "${trace_count}" != "8" ]]; then
    echo "[ERROR] Expected 8 trace files, but found ${trace_count} in ${TRACE_DIR}" >&2
    exit 1
fi

demo_require_file "${TRACE_DIR}/pmpi-trace-rank-0.txt"

demo_info "Generated ${trace_count} trace files successfully"
demo_info "Example trace: ${TRACE_DIR}/pmpi-trace-rank-0.txt"
