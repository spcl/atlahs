#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

demo_prepare_layout
demo_require_dir "${RAW_NSYS_DIR}" "The cached raw Nsight reports are not available."
demo_require_file "${NSYS_TO_SQLITE_SCRIPT}" "The ATLAHS sqlite export helper is missing."

rm -f "${SQLITE_DIR}"/*.sqlite
demo_info "Exporting Nsight reports from ${RAW_NSYS_DIR} into ${SQLITE_DIR}"
bash "${NSYS_TO_SQLITE_SCRIPT}" "${RAW_NSYS_DIR}" "${SQLITE_DIR}"
demo_info "SQLite export finished"
