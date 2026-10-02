#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 RUN_ID CONFIG_JSON" >&2
    exit 2
fi
RUN_ID="$1"
CONFIG="$2"
case "$RUN_ID" in *[!a-zA-Z0-9_.-]*|''|.|..) echo "Invalid run id" >&2; exit 2;; esac
OUTPUT_ROOT="outputs"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
archive_results() {
    local status=$?
    trap - EXIT
    mkdir -p "${OUTPUT_DIR}"
    printf '%s\n' "$status" > "${OUTPUT_DIR}/wrapper_exit_code.txt"
    tar -czf "result_${RUN_ID}.tar.gz" -C "${OUTPUT_ROOT}" "${RUN_ID}"
    exit "$status"
}
trap archive_results EXIT

tar -xzf hep_ssl-code.tar.gz
tar -xzf prepared-data.tar.gz
# Archives contain src/ at their root and prepared/ at their root respectively.
test -f src/train_pairwise.py
test -f prepared/prepared.json
test -f "$CONFIG"
python -u -m src.train_pairwise --config "$CONFIG" --prepared prepared --run-dir "$OUTPUT_DIR"
