#!/bin/bash
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 RUN_ID [pretrained|random]" >&2
    exit 2
fi
RUN_ID="$1"
ENCODER_MODE="${2:-pretrained}"
case "$RUN_ID" in *[!a-zA-Z0-9_.-]*|''|.|..) echo "Invalid run id" >&2; exit 2;; esac
case "$ENCODER_MODE" in pretrained|random) ;; *) echo "Invalid encoder mode" >&2; exit 2;; esac
OUTPUT_ROOT="classifier_output"
OUTPUT_NAME="${RUN_ID}_${ENCODER_MODE}"
OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_NAME}"
archive_results() {
    local status=$?
    trap - EXIT
    mkdir -p "$OUTPUT_DIR"
    printf '%s\n' "$status" > "${OUTPUT_DIR}/wrapper_exit_code.txt"
    tar -czf "result_classifier_${OUTPUT_NAME}.tar.gz" -C "$OUTPUT_ROOT" "$OUTPUT_NAME"
    exit "$status"
}
trap archive_results EXIT

tar -xzf hep_ssl-code.tar.gz
tar -xzf prepared-data.tar.gz
tar -xzf "result_${RUN_ID}.tar.gz"
test -f "${RUN_ID}/checkpoints/best.pt"
python -u -m src.evaluate_pairwise --run-dir "$RUN_ID" --prepared prepared \
    --output-dir "$OUTPUT_DIR" --encoder-mode "$ENCODER_MODE"
