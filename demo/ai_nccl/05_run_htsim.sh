#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

demo_prepare_layout
demo_require_file "${BIN_FILE}" "Run 04_run_loggopsim.sh first."
demo_require_file "${HTSIM_EXEC}" "Build htsim_uec first."
demo_require_file "${HTSIM_TOPO}" "The selected HTSIM topology file is missing."

rm -f "${HTSIM_LOG}" "${HTSIM_SUMMARY_JSON}"
demo_info "Running htsim_uec on ${BIN_FILE}"
demo_info "HTSIM topology: ${HTSIM_TOPO}"
demo_info "HTSIM linkspeed (Mbps): ${HTSIM_LINKSPEED}"
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

"${PYTHON_BIN}" - <<'PY' "${HTSIM_LOG}" "${HTSIM_SUMMARY_JSON}" "${HTSIM_TOPO}" "${HTSIM_LINKSPEED}"
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
text = log_path.read_text(errors="ignore")

host_times = {}
for line in text.splitlines():
    m = re.search(r"^Host\s+(\d+):\s+(\d+)\s*$", line)
    if m:
        host_times[int(m.group(1))] = int(m.group(2))

htsim_time_match = re.search(r"It terminates!\s+Htsim time\s+(\d+)", text)
max_host_match = re.search(r"Maximum finishing time at host\s+(\d+):\s+(\d+)", text)

max_host = None
max_host_time = None
if max_host_match:
    max_host = int(max_host_match.group(1))
    max_host_time = int(max_host_match.group(2))
elif host_times:
    max_host, max_host_time = max(host_times.items(), key=lambda item: item[1])

summary = {
    "log_path": str(log_path),
    "host_lines_found": len(host_times),
    "topology": sys.argv[3],
    "linkspeed_mbps": int(sys.argv[4]),
    "htsim_time_ns": int(htsim_time_match.group(1)) if htsim_time_match else None,
    "max_host": max_host,
    "max_host_time_ns": max_host_time,
}
out_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
