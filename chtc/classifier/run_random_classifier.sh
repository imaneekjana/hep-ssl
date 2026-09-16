#!/bin/bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Expected two arguments:" >&2
    echo "OUTPUT_NAME REFERENCE_EXPERIMENT" >&2
    exit 2
fi

OUTPUT_NAME="$1"
REFERENCE_EXPERIMENT="$2"

REFERENCE_ARCHIVE="result_${REFERENCE_EXPERIMENT}.tar.gz"
REFERENCE_DIR="${REFERENCE_EXPERIMENT}"

OUTPUT_ROOT="classifier_output"
OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_NAME}"

RESULT_ARCHIVE="result_classifier_${OUTPUT_NAME}.tar.gz"

export MPLBACKEND=Agg

echo "=================================================="
echo "Classifier experiment: ${OUTPUT_NAME}"
echo "Encoder mode: random frozen encoder"
echo "Reference experiment: ${REFERENCE_EXPERIMENT}"
echo "Random encoder seed: 42"
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

if [ ! -f "${REFERENCE_ARCHIVE}" ]; then
    echo "Missing reference archive: ${REFERENCE_ARCHIVE}" >&2
    exit 5
fi

echo "Extracting source code..."
tar -xzf hep_ssl-code.tar.gz

echo "Extracting ColliderML data..."
tar -xzf colliderml-data-pairwise-2500.tar.gz

echo "Extracting reference experiment..."
tar -xzf "${REFERENCE_ARCHIVE}"

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

if [ ! -f "${REFERENCE_DIR}/args.json" ]; then
    echo "Missing reference args.json." >&2
    exit 7
fi

if [ ! -f "${REFERENCE_DIR}/stats.npy" ]; then
    echo "Missing reference stats.npy." >&2
    exit 8
fi

if [ ! -f "src/evaluation/run_pairwise_random_classifier.py" ]; then
    echo "Missing random classifier entry point." >&2
    exit 9
fi

if [ ! -f "src/evaluation/classification.py" ]; then
    echo "Missing Aneek classification module." >&2
    exit 10
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

echo "Reference directory: ${REFERENCE_DIR}"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

python -u src/evaluation/run_pairwise_random_classifier.py \
    --pretraining-dir "${REFERENCE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --encoder-batch-size 64 \
    --train-fraction 0.8 \
    --classifier-random-state 42 \
    --random-encoder-seed 42 \
    --classifiers \
        logistic_regression \
        knn \
        gradient_boosting

if [ ! -f "${OUTPUT_DIR}/summary.json" ]; then
    echo "Classifier finished without producing summary.json." >&2
    exit 11
fi

echo "Creating result archive..."

tar -czf "${RESULT_ARCHIVE}" \
    -C "${OUTPUT_ROOT}" \
    "${OUTPUT_NAME}"

echo "Created: ${RESULT_ARCHIVE}"
ls -lh "${RESULT_ARCHIVE}"

echo "Random classifier experiment completed successfully."