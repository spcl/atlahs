#!/usr/bin/env bash

set -euo pipefail

# Stage 2 of the demo:
# 1. Convert the PMPI traces into a GOAL schedule with schedgen.
# 2. Convert the GOAL text schedule into LogGOPSim's binary format with txt2bin.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

demo_require_file "${SCHEDGEN_EXEC}" "Build schedgen first in goal_gen/hpc/Schedgen/."
demo_require_file "${TXT2BIN_EXEC}" "Build txt2bin first in sim/LogGOPSim/."
demo_require_file "${TRACE_DIR}/pmpi-trace-rank-0.txt" "Run demo/01_trace_lulesh.sh first."

rm -f "${GOAL_FILE}" "${BIN_FILE}"

demo_info "Generating GOAL schedule from ${TRACE_DIR}/pmpi-trace-rank-0.txt"
"${SCHEDGEN_EXEC}" \
    -p trace \
    --traces "${TRACE_DIR}/pmpi-trace-rank-0.txt" \
    -o "${GOAL_FILE}"

demo_require_file "${GOAL_FILE}"

demo_info "Converting GOAL text to LogGOPSim binary format"
"${TXT2BIN_EXEC}" \
    -i "${GOAL_FILE}" \
    -o "${BIN_FILE}"

demo_require_file "${BIN_FILE}"

demo_info "Generated GOAL file: ${GOAL_FILE}"
demo_info "Generated binary file: ${BIN_FILE}"
