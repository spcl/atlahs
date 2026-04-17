#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

demo_prepare_layout
demo_require_file "${NCCL_V2_MAIN}" "Initialize the nccl_generator_v2 submodule first."
demo_require_file "${NPKIT_SIMPLE}"
demo_require_file "${NPKIT_LL}"
demo_require_dir "${SQLITE_DIR}" "Run 02_export_sqlite.sh first."

rm -f "${GOAL_DIR}"/rank_*.goal "${GOAL_DIR}/output.goal"
demo_info "Generating GOAL with nccl_generator_v2 for ${CASE}"
(
    cd "${NCCL_V2_ROOT}"
    PYTHONUNBUFFERED=1 NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}" \
        "${PYTHON_BIN}" "${NCCL_V2_MAIN}" \
        --trace_dir "${SQLITE_DIR}" \
        --output_dir "${GOAL_DIR}" \
        --npkit_data_simple "${NPKIT_SIMPLE}" \
        --npkit_data_ll "${NPKIT_LL}" \
        --parallel_generation \
        --concatenate \
        --delete_parts \
        --n_workers "${NCCL_V2_WORKERS}"
)
demo_require_file "${GOAL_FILE}" "V2 goal generation did not produce output.goal."
demo_info "Generated ${GOAL_FILE}"
