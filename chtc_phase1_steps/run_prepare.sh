#!/usr/bin/env bash
# One real pair per CPU job; outputs and receipt are bound by SHA256.
set -euo pipefail
if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
    echo "Usage: $0 PAIR_ID CONFIG_BASENAME INPUT_ARCHIVE PREPARED_ARCHIVE RECEIPT_BASENAME [prepare|verify]" >&2
    exit 2
fi
PAIR_ID="$1"
CONFIG_BASENAME="$2"
RAW_ARCHIVE="$3"
PREPARED_ARCHIVE="$4"
RECEIPT_BASENAME="$5"
PREPARATION_MODE="${6:-prepare}"
if [ "$PREPARATION_MODE" != prepare ] && [ "$PREPARATION_MODE" != verify ]; then
    echo "Mode must be prepare or verify" >&2
    exit 2
fi
for transfer_name in "$PAIR_ID" "$CONFIG_BASENAME" "$RAW_ARCHIVE" "$PREPARED_ARCHIVE" "$RECEIPT_BASENAME"; do
    if [[ ! "$transfer_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "All arguments must be plain transfer-file basenames: $transfer_name" >&2
        exit 2
    fi
done
finish() {
    finish_rc=$?
    trap - EXIT
    set +e
    python - "$finish_rc" "$PAIR_ID" "$CONFIG_BASENAME" "$RAW_ARCHIVE" "$PREPARED_ARCHIVE" "$RECEIPT_BASENAME" "$PREPARATION_MODE" <<'PY'
import json, os, sys, tarfile
from pathlib import Path
from runtime_utils import sha256_file, write_json
rc = int(sys.argv[1])
pair_id, config, raw_archive, archive_name, receipt_name, mode = sys.argv[2:]
archive, receipt = Path(archive_name), Path(receipt_name)
try:
    details = json.loads(receipt.read_text()) if receipt.is_file() else {}
except (OSError, ValueError):
    details = {}
valid_receipt = (details.get('success') is True and details.get('synthetic') is False
                 and details.get('pair_id') == pair_id and details.get('archive') == archive_name
                 and archive.is_file() and details.get('archive_sha256') == sha256_file(archive)
                 and details.get('archive_bytes') == archive.stat().st_size
                 and Path(config).is_file() and details.get('config_sha256') == sha256_file(config)
                 and Path('hep_ssl-code.tar.gz').is_file()
                 and details.get('code_archive_sha256') == sha256_file('hep_ssl-code.tar.gz'))
if rc == 0 and not valid_receipt:
    rc = 1
    details['error'] = 'Preparation returned without a complete, digest-matching real-data receipt.'
if rc != 0:
    details.update(schema_version=1, stage='prepare', pair_id=pair_id, success=False, synthetic=False,
                   archive=archive_name, config=config, raw_archive=None if mode == 'verify' else raw_archive, exit_code=rc,
                   provenance_kind='verified_existing_prepared' if mode == 'verify' else 'new_preparation')
    details.setdefault('error', 'Preparation failed; inspect the returned .err/.out and this diagnostic archive.')
    for field, path in (('config_sha256', config), ('code_archive_sha256', 'hep_ssl-code.tar.gz')):
        if Path(path).is_file():
            details[field] = sha256_file(path)
    if 'code_archive_sha256' in details:
        details['source_archive_sha256'] = details['code_archive_sha256']
    failure = Path('failed_preparation_' + pair_id)
    failure.mkdir(exist_ok=True)
    write_json(failure / 'FAILED.json', details)
    if mode == 'prepare':
        # Failed preparation returns diagnostics instead of a partial prepared
        # payload. Verification preserves its input archive on every outcome.
        temporary = archive.with_name(archive.name + '.failure-partial')
        with tarfile.open(temporary, 'w:gz') as bundle:
            bundle.add(failure, arcname=failure.name)
        os.replace(temporary, archive)
    details.update(archive_sha256=sha256_file(archive) if archive.is_file() else None,
                   archive_bytes=archive.stat().st_size if archive.is_file() else None,
                   prepared_archive_bytes=archive.stat().st_size if archive.is_file() else None)
    write_json(receipt, details)
status = {'schema_version': 1, 'stage': 'prepare', 'pair_id': pair_id, 'exit_code': rc,
          'success': rc == 0, 'synthetic': False, 'archive': archive_name,
          'archive_sha256': sha256_file(archive) if archive.is_file() else None,
          'archive_bytes': archive.stat().st_size if archive.is_file() else None,
          'receipt': receipt_name, 'receipt_sha256': sha256_file(receipt),
          'config_sha256': details.get('config_sha256'),
          'code_archive_sha256': details.get('code_archive_sha256'),
          'provenance_kind': details.get('provenance_kind')}
write_json('prepare_' + pair_id + '_status.json', status)
print(json.dumps(status), flush=True)
raise SystemExit(rc)
PY
    finalize_rc=$?
    if [ "$finish_rc" -eq 0 ]; then finish_rc="$finalize_rc"; fi
    exit "$finish_rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'Preparation pair: %s\nHost: %s\nWorking directory: %s\n' "$PAIR_ID" "$(hostname)" "$PWD"
test -f "$RAW_ARCHIVE"
test -f "$CONFIG_BASENAME"
if [ "$PREPARATION_MODE" = prepare ]; then test -f input_paths.json; fi
if [ "$PREPARATION_MODE" = verify ] && [ "$RAW_ARCHIVE" != "$PREPARED_ARCHIVE" ]; then
    echo "Verification must preserve the input archive basename" >&2
    exit 2
fi
python runtime_utils.py extract hep_ssl-code.tar.gz .
python - <<'PY'
import sys, numpy, polars
print('Python:', sys.version, flush=True)
print('NumPy:', numpy.__version__, 'Polars:', polars.__version__, flush=True)
PY
prepare_command=(python -u prepare_data.py --pair-id "$PAIR_ID" --raw-root raw --archive "$RAW_ARCHIVE"
    --config "$CONFIG_BASENAME" --input-paths input_paths.json
    --prepared-archive "$PREPARED_ARCHIVE" --receipt "$RECEIPT_BASENAME")
if [ "$PREPARATION_MODE" = prepare ]; then
    python runtime_utils.py extract "$RAW_ARCHIVE" raw
else
    prepare_command+=(--verify-existing)
fi
"${prepare_command[@]}"
