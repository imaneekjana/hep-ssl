#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Expected one argument: EXPERIMENT" >&2
    exit 2
fi

EXPERIMENT="$1"

PRETRAINING_ARCHIVE="result_${EXPERIMENT}.tar.gz"
PRETRAINING_DIR="${EXPERIMENT}"

OUTPUT_ROOT="classifier_output"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT}"

RESULT_ARCHIVE="result_classifier_${EXPERIMENT}.tar.gz"

export MPLBACKEND=Agg

echo "=================================================="
echo "Classifier experiment: ${EXPERIMENT}"
echo "Encoder mode: pretrained"
echo "Running on: $(hostname)"
echo "Working directory: ${PWD}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "=================================================="

if [ ! -f "hep_ssl-code.tar.gz" ]; then
    echo "Missing hep_ssl-code.tar.gz" >&2
    exit 3
fi

if [ ! -f "colliderml-data-pairwise-2500.tar.gz" ]; then
    echo "Missing ColliderML data archive" >&2
    exit 4
fi

if [ ! -f "${PRETRAINING_ARCHIVE}" ]; then
    echo "Missing pretraining archive: ${PRETRAINING_ARCHIVE}" >&2
    exit 5
fi

echo "Extracting source code..."
tar -xzf hep_ssl-code.tar.gz

echo "Extracting ColliderML data..."
tar -xzf colliderml-data-pairwise-2500.tar.gz

echo "Extracting pretrained experiment..."
tar -xzf "${PRETRAINING_ARCHIVE}"

DATA_DIR=$(find "${PWD}" \
    -maxdepth 3 \
    -type d \
    -name 'colliderml-data*' \
    -print \
    -quit)

if [ -z "${DATA_DIR}" ]; then
    echo "Could not find extracted ColliderML data directory." >&2
    exit 6
fi

if [ ! -f "${PRETRAINING_DIR}/checkpoints/best.pt" ]; then
    echo "Missing best checkpoint." >&2
    exit 7
fi

if [ ! -f "${PRETRAINING_DIR}/args.json" ]; then
    echo "Missing args.json." >&2
    exit 8
fi

if [ ! -f "${PRETRAINING_DIR}/stats.npy" ]; then
    echo "Missing stats.npy." >&2
    exit 9
fi

if [ ! -f "src/evaluation/run_pairwise_classifier.py" ]; then
    echo "Missing pretrained classifier entry point." >&2
    exit 10
fi

if [ ! -f "src/evaluation/classification.py" ]; then
    echo "Missing Aneek classification module." >&2
    exit 11
fi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the assigned GPU.")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Reference directory: ${PRETRAINING_DIR}"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

python -u src/evaluation/run_pairwise_classifier.py \
    --pretraining-dir "${PRETRAINING_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --encoder-batch-size 64 \
    --train-fraction 0.8 \
    --classifier-random-state 42 \
    --classifiers \
        logistic_regression \
        knn \
        gradient_boosting

if [ ! -f "${OUTPUT_DIR}/summary.json" ]; then
    echo "Classifier finished without producing summary.json." >&2
    exit 12
fi

echo "Creating result archive..."

tar -czf "${RESULT_ARCHIVE}" \
    -C "${OUTPUT_ROOT}" \
    "${EXPERIMENT}"

echo "Created: ${RESULT_ARCHIVE}"
ls -lh "${RESULT_ARCHIVE}"

echo "Pretrained classifier experiment completed successfully."