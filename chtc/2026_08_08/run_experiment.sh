#!/bin/bash

set -euo pipefail

EXP_NAME="$1"
AUG_ORDER="$2"
SOFT_MAX_FRACTION="$3"
SOFT_CELL_FRACTION="$4"
SHARE_MAX_FRACTION="$5"
LEARNING_RATE="$6"
WEIGHT_DECAY="$7"
BATCH_SIZE="$8"
TEMPERATURE="$9"
SEED="${10}"

echo "=================================================="
echo "Experiment: ${EXP_NAME}"
echo "Running on: $(hostname)"
echo "Working directory: ${PWD}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Augmentation order: ${AUG_ORDER}"
echo "Soft removal budget: ${SOFT_MAX_FRACTION}"
echo "Soft-cell threshold: ${SOFT_CELL_FRACTION}"
echo "Local sharing budget: ${SHARE_MAX_FRACTION}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Weight decay: ${WEIGHT_DECAY}"
echo "Batch size: ${BATCH_SIZE}"
echo "Temperature: ${TEMPERATURE}"
echo "Seed: ${SEED}"
echo "=================================================="

mkdir -p outputs

# Always create an output archive, including when Python exits with an error.
archive_results() {
    local status=$?

    trap - EXIT
    set +e

    if [ ! -d "outputs/${EXP_NAME}" ]; then
        mkdir -p "outputs/${EXP_NAME}"
    fi

    printf "%s\n" "${status}" \
        > "outputs/${EXP_NAME}/wrapper_exit_code.txt"

    tar -czf "result_${EXP_NAME}.tar.gz" \
        -C outputs "${EXP_NAME}"

    exit "${status}"
}

trap archive_results EXIT

echo "Extracting source code..."
tar -xzf hep_ssl-code.tar.gz

echo "Extracting ColliderML data..."
tar -xzf colliderml-data-v1.tar.gz

echo "Checking GPU..."

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the assigned GPU")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Starting formal training..."

python -u src/train_colliderml_planar_physics_v2.py \
    --experiment-name "${EXP_NAME}" \
    --trainevents 1500 \
    --split ttbar_ggf_total3000 \
    --train-augmentation-order "${AUG_ORDER}" \
    --beam-reflection-prob 0.5 \
    --soft-max-fraction "${SOFT_MAX_FRACTION}" \
    --soft-cell-fraction "${SOFT_CELL_FRACTION}" \
    --share-max-fraction "${SHARE_MAX_FRACTION}" \
    --epochs 18 \
    --batch-size "${BATCH_SIZE}" \
    --workers 0 \
    --lr "${LEARNING_RATE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --temperature "${TEMPERATURE}" \
    --hidden-dim 16 \
    --latent-dim 64 \
    --proj-dim 32 \
    --gravnet-k 8 \
    --space-dim 4 \
    --propagate-dim 16 \
    --seed "${SEED}" \
    --data-dir "${PWD}/colliderml-data" \
    --output-dir "${PWD}/outputs"

echo "Experiment ${EXP_NAME} completed successfully."
