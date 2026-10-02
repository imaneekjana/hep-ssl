#!/usr/bin/env bash
# Runs on a GPU execution node. The two jobs keep the original scheduler horizon.
set -euo pipefail
if [ "$#" -ne 3 ]; then echo "Usage: $0 first|continue RUN_ID OUTPUT_ARCHIVE" >&2; exit 2; fi
STAGE="$1"
RUN_ID="$2"
ARCHIVE="$3"
case "$STAGE" in first|continue) ;; *) exit 2;; esac
case "$RUN_ID" in *[!a-zA-Z0-9_.-]*|''|.|..) echo "Invalid run id" >&2; exit 2;; esac
case "$ARCHIVE" in *[!a-zA-Z0-9_.-]*|''|.|..) echo "Invalid archive name" >&2; exit 2;; esac
RUN_DIR="outputs/${RUN_ID}"
finish() {
    status=$?
    trap - EXIT
    set +e
    mkdir -p "$RUN_DIR"
    printf '%s\n' "$status" > "$RUN_DIR/wrapper_exit_code.txt"
    tar -czf "$ARCHIVE" -C outputs "$RUN_ID"
    archive_status=$?
    if [ "$status" -eq 0 ] && [ "$archive_status" -ne 0 ]; then status="$archive_status"; fi
    python - "$STAGE" "$RUN_ID" "$ARCHIVE" "$status" <<'PY'
import json, sys
from pathlib import Path
stage, run_id, archive, code = sys.argv[1:]
record = {'stage': stage, 'run_id': run_id, 'archive': archive,
          'exit_code': int(code), 'success': code == '0'}
details = Path('outputs') / run_id / (stage + '_details.json')
if details.exists():
    record['details'] = json.loads(details.read_text())
Path(stage + '_status.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
print(json.dumps(record), flush=True)
PY
    exit "$status"
}
trap finish EXIT
printf 'GPU job stage: %s\nHost: %s\nWorking directory: %s\n' "$STAGE" "$(hostname)" "$PWD"
tar -xzf hep_ssl-code.tar.gz
tar -xzf prepared-data.tar.gz
test -f prepared/prepared.json
if [ "$STAGE" = continue ]; then
    mkdir outputs
    tar -xzf first_epoch.tar.gz -C outputs
    test -f "$RUN_DIR/checkpoints/last.pt"
    test "$(cat "$RUN_DIR/wrapper_exit_code.txt")" = 0
fi
python -u gpu_worker.py --stage "$STAGE" --run-id "$RUN_ID"
