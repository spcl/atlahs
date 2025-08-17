#!/bin/bash
set -x

## ******************************************************************************
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
## Copyright (c) 2024 Georgia Institute of Technology
## ******************************************************************************

# find the absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="${SCRIPT_DIR:?}/../.."
EXAMPLE_DIR="${PROJECT_DIR:?}/examples/network_analytical"

# paths
ASTRA_SIM="${PROJECT_DIR:?}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
# WORKLOAD="${EXAMPLE_DIR:?}/workload/AllReduce_1MB"
WORKLOAD=$ASTRA_SIM_WORKLOAD
echo "[AstraSim Run Script] AstraSim workload: $WORKLOAD"
# SYSTEM="${EXAMPLE_DIR:?}/system.json"
SYSTEM=$ASTRA_SIM_SYSTEM
echo "[AstraSim Run Script] AstraSim system: $SYSTEM"
# NETWORK="${EXAMPLE_DIR:?}/network.yml"
NETWORK=$ASTRA_SIM_NETWORK
echo "[AstraSim Run Script] AstraSim network: $NETWORK"
REMOTE_MEMORY="${EXAMPLE_DIR:?}/remote_memory.json"
# echo "[AstraSim Run Script] AstraSim remote memory: $REMOTE_MEMORY"

# start
echo "[ASTRA-sim] Compiling ASTRA-sim with the Analytical Network Backend..."
echo ""

# Compile
# "${PROJECT_DIR:?}"/build/astra_analytical/build.sh

echo ""
echo "[ASTRA-sim] Compilation finished."
echo "[ASTRA-sim] Running ASTRA-sim Example with Analytical Network Backend..."
echo ""

# run ASTRA-sim
"${ASTRA_SIM:?}" \
    --workload-configuration="${WORKLOAD}" \
    --system-configuration="${SYSTEM:?}" \
    --remote-memory-configuration="${REMOTE_MEMORY:?}" \
    --network-configuration="${NETWORK:?}"

# finalize
echo ""
echo "[ASTRA-sim] Finished the execution."
