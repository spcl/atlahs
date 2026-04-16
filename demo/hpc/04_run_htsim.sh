#!/usr/bin/env bash

set -euo pipefail

# Stage 4 of the demo:
# Run the same LogGOPSim binary schedule through the HTSIM backend.
#
# The default HTSIM parameters below mirror the style already used in this repo:
# - leaf_spine_tiny.topo because this demo uses 8 ranks
# - 200000 Mbps linkspeed
# - ECMP host routing
# - 4096-byte MTU
# - 128 paths
# - large queue to avoid premature queue bottlenecks in the demo setup

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

demo_require_file "${HTSIM_EXEC}" "Build htsim_uec first in sim/htsim-backend/sim/datacenter/."
demo_require_file "${HTSIM_TOPO}" "The expected 8-node HTSIM topology file is missing."
demo_require_file "${BIN_FILE}" "Run demo/02_generate_goal_and_bin.sh first."

HTSIM_NODES="${HTSIM_NODES:-8}"
HTSIM_LINKSPEED="${HTSIM_LINKSPEED:-200000}"
HTSIM_MTU="${HTSIM_MTU:-4096}"
HTSIM_PATHS="${HTSIM_PATHS:-128}"
HTSIM_QUEUE="${HTSIM_QUEUE:-1000000}"
HTSIM_SEED="${HTSIM_SEED:-4}"

demo_info "Running htsim_uec on ${BIN_FILE}"
demo_info "Writing HTSIM stdout to ${HTSIM_LOG}"
"${HTSIM_EXEC}" \
    -topo "${HTSIM_TOPO}" \
    -goal "${BIN_FILE}" \
    -linkspeed "${HTSIM_LINKSPEED}" \
    -nodes "${HTSIM_NODES}" \
    -strat ecmp_host \
    -mtu "${HTSIM_MTU}" \
    -paths "${HTSIM_PATHS}" \
    -lgs_flow_stats \
    -q "${HTSIM_QUEUE}" \
    -seed "${HTSIM_SEED}" > "${HTSIM_LOG}"

demo_info "HTSIM run finished"
demo_info "Inspect the saved output in ${HTSIM_LOG}"
