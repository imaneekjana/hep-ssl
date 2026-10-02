#!/bin/bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 RUN_ID" >&2
    exit 2
fi
exec bash "$(dirname "$0")/run_classifier.sh" "$1" random
