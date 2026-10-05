#!/usr/bin/env bash
# Runs on an allocated execution node, using the original training source tar.
set -euo pipefail
if [ "$#" -ne 5 ]; then
    echo "Usage: RUN_ID PREPARED_ARCHIVE RESULT_ARCHIVE RECEIPT DEVICE" >&2
    exit 2
fi
RUN_ID="$1"; PREPARED="$2"; INPUT_RESULT="$3"; RECEIPT="$4"; DEVICE="$5"
for value in "$RUN_ID" "$PREPARED" "$INPUT_RESULT" "$RECEIPT"; do
    case "$value" in *[!a-zA-Z0-9_.-]*|''|.|..|-*) echo "Invalid basename: $value" >&2; exit 2;; esac
done
case "$DEVICE" in cuda|cpu) ;; *) exit 2;; esac
OUT="classifier_output/$RUN_ID"
ARCHIVE="classifier_${RUN_ID}.tar.gz"
STATUS="classifier_status_${RUN_ID}.json"
WORKER_LOG="classifier_${RUN_ID}.log"
finish() {
    rc=$?
    trap - EXIT
    set +e
    mkdir -p "$OUT"
    [ ! -f "$WORKER_LOG" ] || cp "$WORKER_LOG" "$OUT/wrapper.log"
    if [ "$rc" -eq 0 ] && [ ! -f "$OUT/metrics.json" ]; then rc=1; fi
    printf '%s\n' "$rc" > "$OUT/wrapper_exit_code.txt"
    tar -czf "$ARCHIVE" -C classifier_output "$RUN_ID"
    archive_rc=$?
    if [ "$rc" -eq 0 ] && [ "$archive_rc" -ne 0 ]; then rc=$archive_rc; fi
    python - "$RUN_ID" "$OUT" "$ARCHIVE" "$STATUS" "$rc" "$archive_rc" <<'PY'
import json, sys
from pathlib import Path
run, out, archive, status, code, archive_code = sys.argv[1:]
record = {'run_id': run, 'exit_code': int(code), 'archive_exit_code': int(archive_code),
          'success': code == '0' and archive_code == '0', 'archive': archive,
          'checkpoint': 'best.pt', 'encoder_mode': 'pretrained'}
metrics = Path(out) / 'metrics.json'
if metrics.is_file():
    data = json.loads(metrics.read_text())
    record['classification'] = data['classification']
Path(status).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
print(json.dumps(record), flush=True)
PY
    status_rc=$?
    if [ "$rc" -eq 0 ] && [ "$status_rc" -ne 0 ]; then rc=$status_rc; fi
    exit "$rc"
}
trap finish EXIT
run_logged() { "$@" 2>&1 | tee -a "$WORKER_LOG"; }
printf 'Run: %s\nHost: %s\nWorking directory: %s\n' "$RUN_ID" "$(hostname)" "$PWD" | tee "$WORKER_LOG"
# The source archive is the user's own previously used training code.
run_logged tar -xzf hep_ssl-code.tar.gz
run_logged python - "$PREPARED" "$RECEIPT" "$DEVICE" <<'PY'
import hashlib, json, sys
import torch
archive, receipt_path, device = sys.argv[1:]
receipt = json.load(open(receipt_path))
if receipt.get('success') is not True or receipt.get('synthetic') is not False:
    raise ValueError('A successful real-data prepared receipt is required.')
h = hashlib.sha256()
with open(archive, 'rb') as f:
    for block in iter(lambda: f.read(1024 * 1024), b''):
        h.update(block)
if h.hexdigest() != receipt['archive_sha256']:
    raise ValueError('Prepared archive differs from the pretraining data receipt.')
if device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA requested but unavailable; no CPU fallback.')
print('PyTorch:', torch.__version__, 'device:', device, flush=True)
PY
run_logged python chtc_phase1_steps/runtime_utils.py extract "$PREPARED" . --expected-root prepared
run_logged python chtc_phase1_steps/runtime_utils.py extract "$INPUT_RESULT" pretraining --expected-root "$RUN_ID"
# This entry point freezes the encoder, uses CleanDataset and the saved split,
# fits classifiers/probes on train only and evaluates on held-out partitions.
run_logged python -u -m src.evaluate_pairwise \
    --run-dir "pretraining/$RUN_ID" \
    --checkpoint "pretraining/$RUN_ID/checkpoints/best.pt" \
    --prepared prepared \
    --output-dir "$OUT" \
    --encoder-mode pretrained \
    --device "$DEVICE" \
    --batch-size 64 --num-workers 0 \
    --classifier-c 1.0 --probe-alpha 1.0 --seed 42
