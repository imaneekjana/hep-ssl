#!/bin/bash

set -euo pipefail

OLD_ARCHIVE="colliderml-data-v1.tar.gz"
NEW_ARCHIVE="colliderml-data-pairwise-2500.tar.gz"

echo "Running on: $(hostname)"
echo "Working directory: ${PWD}"

if [ ! -f "${OLD_ARCHIVE}" ]; then
    echo "Missing input archive: ${OLD_ARCHIVE}" >&2
    exit 2
fi

echo "Extracting existing ColliderML cache..."
tar -xzf "${OLD_ARCHIVE}"

DATASET_ROOT=$(find "${PWD}" \
    -maxdepth 4 \
    -type d \
    -name 'CERN__ColliderML-Release-1' \
    -print \
    -quit)

if [ -z "${DATASET_ROOT}" ]; then
    echo "Could not locate CERN__ColliderML-Release-1 after extraction." >&2
    exit 3
fi

DATA_DIR=$(dirname "${DATASET_ROOT}")

echo "Reusing cache directory: ${DATA_DIR}"
echo "Extending/verifying three channels at 2500 events each..."

python -u download_data_pairwise_2500.py \
    --data-dir "${DATA_DIR}" \
    --events-per-channel 2500

echo "Creating new archive..."
tar -czf "${NEW_ARCHIVE}" \
    -C "$(dirname "${DATA_DIR}")" \
    "$(basename "${DATA_DIR}")"

echo "Created: ${NEW_ARCHIVE}"
ls -lh "${NEW_ARCHIVE}"