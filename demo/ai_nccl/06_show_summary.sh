#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

demo_prepare_layout

demo_info "Case: ${CASE} (${CASE_LABEL})"
demo_info "Reference case root: ${REF_CASE_ROOT}"
demo_info "Cached trace dir: ${RAW_NSYS_DIR}"
demo_info "Live sqlite dir: ${SQLITE_DIR}"
demo_info "Live V2 goal: ${GOAL_FILE}"
demo_info "Live bin: ${BIN_FILE}"
demo_info "Live LGS log: ${LGS_LOG}"
demo_info "Live HTSIM log: ${HTSIM_LOG}"
demo_info "Hardware runtime (reference): ${HARDWARE_RUNTIME_NS} ns"
demo_info "LGS parameters: -L ${LGS_L} -o ${LGS_o} -g ${LGS_g} -O ${LGS_O} -G ${LGS_G} -S ${LGS_S}"
demo_info "HTSIM topology: ${HTSIM_TOPO}"
demo_info "HTSIM linkspeed (Mbps): ${HTSIM_LINKSPEED}"

echo
"${PYTHON_BIN}" - <<'PY' "${LGS_LOG}" "${HTSIM_SUMMARY_JSON}" "${HARDWARE_RUNTIME_NS}"
import json
import re
import sys
from pathlib import Path

lgs_log = Path(sys.argv[1])
htsim_json = Path(sys.argv[2])
hardware_ns = int(sys.argv[3])

def fmt_delta(runtime_ns, hardware_ns):
    if runtime_ns is None:
        return "None"
    delta = runtime_ns - hardware_ns
    pct = (delta / hardware_ns) * 100 if hardware_ns else 0.0
    return f"{delta} ns ({pct:+.2f}%)"

lgs_ns = None
if lgs_log.exists():
    text = lgs_log.read_text(errors="ignore")
    host_times = [int(m.group(2)) for m in re.finditer(r"^Host\s+(\d+):\s+(\d+)\s*$", text, re.MULTILINE)]
    if host_times:
        lgs_ns = max(host_times)

htsim = {}
if htsim_json.exists():
    htsim = json.loads(htsim_json.read_text())
htsim_ns = htsim.get("max_host_time_ns") or htsim.get("htsim_time_ns")

print(f"HW runtime     : {hardware_ns} ns")
print(f"LGS runtime    : {lgs_ns} ns")
print(f"HTSIM runtime  : {htsim_ns} ns")
print(f"LGS - HW       : {fmt_delta(lgs_ns, hardware_ns)}")
print(f"HTSIM - HW     : {fmt_delta(htsim_ns, hardware_ns)}")
PY

if [[ -f "${REFERENCE_LINK}/results/final_compare_stage_motif/${CASE}_stage_motif_with_stock_v2_micro_1ms.png" ]]; then
    echo
    echo "Reference plot: ${REFERENCE_LINK}/results/final_compare_stage_motif/${CASE}_stage_motif_with_stock_v2_micro_1ms.png"
fi
