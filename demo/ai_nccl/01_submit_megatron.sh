#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

demo_prepare_layout
demo_require_file "${JOB_SCRIPT}" "The 2-node sbatch wrapper is missing."

SUBMIT_MODE="${SUBMIT_MODE:-test-only}"
case "${SUBMIT_MODE}" in
    print)
        demo_info "Prepared job script: ${JOB_SCRIPT}"
        echo "sbatch ${JOB_SCRIPT}"
        ;;
    test-only)
        demo_info "Validating the 2-node Megatron trace job with --test-only"
        sbatch --test-only "${JOB_SCRIPT}"
        ;;
    submit)
        demo_info "Submitting the 2-node Megatron trace job"
        sbatch "${JOB_SCRIPT}"
        ;;
    *)
        echo "[ERROR] Unsupported SUBMIT_MODE=${SUBMIT_MODE}. Use print, test-only, or submit." >&2
        exit 1
        ;;
esac
