#!/usr/bin/env bash
# One isolated execution scratch directory per run/attempt; no submission here.
set -euo pipefail
if [ "$#" -lt 4 ] || [ "$#" -gt 6 ]; then
    echo "Usage: $0 RUN_ID CONFIG_BASENAME PREPARED_ARCHIVE RECEIPT_BASENAME [STOP_AFTER_EPOCH=0] [RESUME_BASENAME=-]" >&2
    exit 2
fi
RUN_ID="$1"
CONFIG="$2"
PREPARED_ARCHIVE="$3"
RECEIPT="$4"
STOP_AFTER_EPOCH="${5:-0}"
RESUME_ARCHIVE="${6:--}"
for value in "$RUN_ID" "$CONFIG" "$PREPARED_ARCHIVE" "$RECEIPT"; do
    case "$value" in *[!a-zA-Z0-9_.-]*|''|.|..|-*) echo "Invalid basename: $value" >&2; exit 2;; esac
done
if [ "$RESUME_ARCHIVE" != '-' ]; then
    case "$RESUME_ARCHIVE" in *[!a-zA-Z0-9_.-]*|''|.|..|-*) echo "Invalid resume basename" >&2; exit 2;; esac
fi
case "$STOP_AFTER_EPOCH" in *[!0-9]*|'') echo "Stopping epoch must be a nonnegative integer" >&2; exit 2;; esac
RUN_DIR="outputs/${RUN_ID}"
ARCHIVE="result_${RUN_ID}.tar.gz"
STATUS_FILE="status_${RUN_ID}.json"
WORKER_LOG="gpu_${RUN_ID}.log"
WRAPPER_STAGE="initialization"

finish() {
    status=$?
    trap - EXIT
    set +e
    mkdir -p "$RUN_DIR"
    mkdir_status=$?
    if [ "$mkdir_status" -ne 0 ] && [ "$status" -eq 0 ]; then status="$mkdir_status"; fi
    if [ -f "$WORKER_LOG" ]; then cp "$WORKER_LOG" "$RUN_DIR/wrapper.log"; fi
    printf '%s\n' "$status" > "$RUN_DIR/wrapper_exit_code.txt"
    python - "$RUN_ID" "$CONFIG" "$ARCHIVE" "$STATUS_FILE" "$status" "$WRAPPER_STAGE" <<'PY'
import json, sys
from pathlib import Path
run_id, config, archive, status_file, code, stage = sys.argv[1:]
details_path = Path('outputs') / run_id / 'job_details.json'
details = {}
try:
    if details_path.exists():
        details = json.loads(details_path.read_text())
except Exception as error:
    details = {'error': f'Cannot read worker diagnostics: {error}'}
if not details:
    details = {'run_id': run_id, 'success': False, 'stage': stage,
               'error': f'Wrapper exited during {stage} with code {code}; see wrapper.log.'}
    details_path.write_text(json.dumps(details, indent=2, allow_nan=False) + '\n')
record = {'run_id': run_id, 'config_basename': config, 'archive': archive,
          'exit_code': int(code), 'success': int(code) == 0 and details.get('success') is True,
          'wrapper_stage': stage, 'pair_id': details.get('pair_id'),
          'completed_epochs': details.get('completed_epochs', 0),
          'configured_total_epochs': details.get('configured_total_epochs'),
          'prepared_fingerprint': details.get('prepared_fingerprint'), 'details': details}
Path(status_file).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
PY
    status_write=$?
    if [ "$status_write" -ne 0 ] && [ "$status" -eq 0 ]; then status="$status_write"; fi
    tar -czf "$ARCHIVE" -C outputs "$RUN_ID"
    archive_status=$?
    if [ "$archive_status" -ne 0 ] && [ "$status" -eq 0 ]; then status="$archive_status"; fi
    python - "$STATUS_FILE" "$status" "$archive_status" <<'PY'
import json, sys
from pathlib import Path
path, code, archive_code = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
record = json.loads(path.read_text())
record.update(exit_code=code, archive_exit_code=archive_code,
              success=code == 0 and archive_code == 0 and record.get('success') is True)
path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
print(json.dumps(record), flush=True)
PY
    final_status_write=$?
    if [ "$final_status_write" -ne 0 ] && [ "$status" -eq 0 ]; then status="$final_status_write"; fi
    exit "$status"
}
trap finish EXIT

run_logged() { "$@" 2>&1 | tee -a "$WORKER_LOG"; }
printf 'GPU run: %s\nHost: %s\nWorking directory: %s\n' "$RUN_ID" "$(hostname)" "$PWD" | tee "$WORKER_LOG"
WRAPPER_STAGE="extract_code"
run_logged python runtime_utils.py extract hep_ssl-code.tar.gz .
WRAPPER_STAGE="worker"
run_logged python -u gpu_worker.py --run-id "$RUN_ID" --config "$CONFIG" \
    --prepared-archive "$PREPARED_ARCHIVE" --receipt "$RECEIPT" \
    --stop-after-epoch "$STOP_AFTER_EPOCH" --resume-archive "$RESUME_ARCHIVE"
WRAPPER_STAGE="complete"
