#!/bin/bash

set -e

EXPERIMENT="$1"

echo "Running linear probe for: ${EXPERIMENT}"

tar -xzf hep_ssl-code.tar.gz
tar -xzf colliderml-data-v1.tar.gz
tar -xzf "result_${EXPERIMENT}.tar.gz"

CHECKPOINT=$(find . \
    -type f \
    -path "*/checkpoints/best.pt" \
    | head -n 1)

DATA_DIR=$(find . \
    -maxdepth 3 \
    -type d \
    -name "colliderml-data*" \
    | head -n 1)

OUTPUT_DIR="classifier_output/${EXPERIMENT}"

mkdir -p "${OUTPUT_DIR}"

echo "Experiment: ${EXPERIMENT}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

python chtc/notebook/classifier/linear_probe.py \
    --checkpoint "${CHECKPOINT}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --events-per-class 1500 \
    --encoder-batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --seed 53

tar -czf \
    "result_linear_probe_${EXPERIMENT}.tar.gz" \
    "classifier_output/${EXPERIMENT}"

echo "Completed: ${EXPERIMENT}"