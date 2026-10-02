#!/usr/bin/env bash
# Runs on a CPU execution node inside the existing container.
set -euo pipefail
if [ "$#" -ne 1 ]; then echo "Usage: $0 RAW_ARCHIVE" >&2; exit 2; fi
RAW_ARCHIVE="$1"
finish() {
    status=$?
    trap - EXIT
    set +e
    if [ ! -f prepared-data.tar.gz ]; then
        mkdir -p failed_preparation
        printf '%s\n' 'Preparation failed; this archive is NOT prepared data.' > failed_preparation/FAILED.txt
        tar -czf prepared-data.tar.gz failed_preparation
    fi
    if [ ! -f prepare_details.json ]; then printf '{}\n' > prepare_details.json; fi
    python - "$status" <<'PY'
import json, sys
from pathlib import Path
p = Path('prepared-data.tar.gz')
record = {'stage': 'prepare', 'exit_code': int(sys.argv[1]), 'success': sys.argv[1] == '0',
          'prepared_archive_bytes': p.stat().st_size if p.exists() else None}
Path('prepare_status.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
PY
    exit "$status"
}
trap finish EXIT
printf 'Preparation host: %s\nWorking directory: %s\n' "$(hostname)" "$PWD"
test -f "$RAW_ARCHIVE"
tar -xzf hep_ssl-code.tar.gz
# Fail before expensive extraction if the preparation runtime lacks dependencies.
python - <<'PY'
import sys, numpy, polars
print('Python:', sys.version, flush=True)
print('NumPy:', numpy.__version__, 'Polars:', polars.__version__, flush=True)
PY
mkdir raw
tar -xzf "$RAW_ARCHIVE" -C raw
python -u prepare_data.py --raw-root raw --archive "$RAW_ARCHIVE" --config pairwise_chtc.json --input-paths input_paths.json
# Newly prepared data has the exact top-level directory expected by training.
tar -czf prepared-data.tar.gz prepared
python - <<'PY'
import json
from pathlib import Path
path = Path('prepare_details.json')
d = json.loads(path.read_text())
d['prepared_archive_bytes'] = Path('prepared-data.tar.gz').stat().st_size
path.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n')
print('PREPARATION_COMPLETE', flush=True)
print('prepared archive bytes:', d['prepared_archive_bytes'], flush=True)
PY
