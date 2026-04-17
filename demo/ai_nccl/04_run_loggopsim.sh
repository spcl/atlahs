#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

demo_prepare_layout
demo_require_file "${GOAL_FILE}" "Run 03_generate_goal_v2.sh first."
demo_require_file "${TXT2BIN_EXEC}" "Build LogGOPSim first."
demo_require_file "${LOGGOPSIM_EXEC}" "Build LogGOPSim first."

rm -f "${BIN_FILE}" "${LGS_LOG}"
demo_info "Converting ${GOAL_FILE} to ${BIN_FILE}"
"${TXT2BIN_EXEC}" -i "${GOAL_FILE}" -o "${BIN_FILE}"

demo_info "Running LogGOPSim on ${BIN_FILE}"
"${LOGGOPSIM_EXEC}" \
    -f "${BIN_FILE}" \
    -L "${LGS_L}" \
    -o "${LGS_o}" \
    -g "${LGS_g}" \
    -O "${LGS_O}" \
    -G "${LGS_G}" \
    -S "${LGS_S}" | tee "${LGS_LOG}"
demo_info "Saved LogGOPSim output to ${LGS_LOG}"
