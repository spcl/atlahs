#!/usr/bin/env bash

set -euo pipefail

# Stage 3 of the demo:
# Run the generated .bin file through LogGOPSim.
#
# These LogGOPS parameters match the values used elsewhere in this repository
# for ATLAHS runs and in ATLAHS_PIPELINE_SUMMARY.md.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

demo_require_file "${LOGGOPSIM_EXEC}" "Build LogGOPSim first in sim/LogGOPSim/."
demo_require_file "${BIN_FILE}" "Run demo/02_generate_goal_and_bin.sh first."

LGS_L="${LGS_L:-3700}"
LGS_o="${LGS_o:-200}"
LGS_g="${LGS_g:-5}"
LGS_G="${LGS_G:-0.04}"
LGS_O="${LGS_O:-0}"
LGS_S="${LGS_S:-0}"

demo_info "Running LogGOPSim on ${BIN_FILE}"
demo_info "Saving a copy of the simulator output to ${LGS_LOG}"
"${LOGGOPSIM_EXEC}" \
    -f "${BIN_FILE}" \
    -L "${LGS_L}" \
    -o "${LGS_o}" \
    -g "${LGS_g}" \
    -G "${LGS_G}" \
    -O "${LGS_O}" \
    -S "${LGS_S}" | tee "${LGS_LOG}"
