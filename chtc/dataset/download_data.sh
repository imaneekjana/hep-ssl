#!/bin/bash

set -euo pipefail

echo "Running on: $(hostname)"
echo "Working directory: $PWD"

export COLLIDERML_DATA_DIR="$PWD/colliderml-data"
export HF_HOME="$PWD/huggingface-cache"

mkdir -p "$COLLIDERML_DATA_DIR"
mkdir -p "$HF_HOME"

for channel in ttbar dihiggs ggf
do
    echo "Downloading channel: ${channel}"

    colliderml download \
        --channels "${channel}" \
        --pileup pu0 \
        --objects calo_hits \
        --max-events 1500

    echo "Finished channel: ${channel}"
done

echo "Creating data archive..."

tar -czf colliderml-data-v1.tar.gz colliderml-data

echo "Data archive created:"
ls -lh colliderml-data-v1.tar.gz
