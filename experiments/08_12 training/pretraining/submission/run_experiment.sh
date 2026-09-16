#!/bin/bash

set -euo pipefail

if [ "$#" -ne 16 ]; then
    echo "Expected 16 arguments, received $#" >&2
    echo "Usage: $0 EXP_NAME CHANNEL_A CHANNEL_B EVENTS_PER_CHANNEL ROTATION_MODE ROTATION ENERGY_NOISE XYZ_NOISE SHIFT_STD CROP_FRACTION AUG_ORDER LEARNING_RATE WEIGHT_DECAY BATCH_SIZE TEMPERATURE SEED" >&2
    exit 2
fi

EXP_NAME="$1"
CHANNEL_A="$2"
CHANNEL_B="$3"
EVENTS_PER_CHANNEL="$4"
ROTATION_MODE="$5"
ROTATION="$6"
ENERGY_NOISE="$7"
XYZ_NOISE="$8"
SHIFT_STD="$9"
CROP_FRACTION="${10}"
AUG_ORDER="${11}"
LEARNING_RATE="${12}"
WEIGHT_DECAY="${13}"
BATCH_SIZE="${14}"
TEMPERATURE="${15}"
SEED="${16}"

OUTPUT_ROOT="outputs"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
RESULT_ARCHIVE="result_${EXP_NAME}.tar.gz"

echo "=================================================="
echo "Experiment: ${EXP_NAME}"
echo "Processes: ${CHANNEL_A} vs ${CHANNEL_B}"
echo "Events per channel: ${EVENTS_PER_CHANNEL}"
echo "Running on: $(hostname)"
echo "Working directory: ${PWD}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Rotation mode: ${ROTATION_MODE}"
echo "Rotation: ${ROTATION}"
echo "Energy noise: ${ENERGY_NOISE} GeV"
echo "XYZ noise: ${XYZ_NOISE} mm"
echo "Shift std: ${SHIFT_STD} mm"
echo "Crop fraction: ${CROP_FRACTION}"
echo "Augmentation order: ${AUG_ORDER}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Weight decay: ${WEIGHT_DECAY}"
echo "Batch size: ${BATCH_SIZE}"
echo "Temperature: ${TEMPERATURE}"
echo "Seed: ${SEED}"
echo "=================================================="

# Always return a result archive, even when training fails. This preserves
# args, partial histories, and the wrapper exit code for diagnosis.
archive_results() {
    local status=$?

    trap - EXIT
    set +e

    mkdir -p "${OUTPUT_DIR}"
    printf "%s\n" "${status}" > "${OUTPUT_DIR}/wrapper_exit_code.txt"

    tar -czf "${RESULT_ARCHIVE}" \
        -C "${OUTPUT_ROOT}" "${EXP_NAME}"

    exit "${status}"
}

trap archive_results EXIT

echo "Extracting source code..."
tar -xzf hep_ssl-code.tar.gz

echo "Extracting ColliderML data..."
tar -xzf colliderml-data-pairwise-2500.tar.gz

DATA_DIR=$(find "${PWD}" \
    -maxdepth 3 \
    -type d \
    -name 'colliderml-data*' \
    -print \
    -quit)

if [ -z "${DATA_DIR}" ]; then
    echo "Could not find the extracted ColliderML data directory." >&2
    exit 3
fi

TRAIN_SCRIPT="src/train_colliderml_planar_pairwise.py"

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "Training entry point not found: ${TRAIN_SCRIPT}" >&2
    exit 4
fi

echo "Training script: ${TRAIN_SCRIPT}"
echo "ColliderML data directory: ${DATA_DIR}"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the assigned GPU")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Starting pretraining..."

python -u "${TRAIN_SCRIPT}" \
    --experiment-name "${EXP_NAME}" \
    --channel-a "${CHANNEL_A}" \
    --channel-b "${CHANNEL_B}" \
    --events-per-channel "${EVENTS_PER_CHANNEL}" \
    --split "${CHANNEL_A}_vs_${CHANNEL_B}_total$((2 * EVENTS_PER_CHANNEL))" \
    --rotation-mode "${ROTATION_MODE}" \
    --rotation "${ROTATION}" \
    --energy-noise "${ENERGY_NOISE}" \
    --xyz-noise "${XYZ_NOISE}" \
    --shift-std "${SHIFT_STD}" \
    --crop-fraction "${CROP_FRACTION}" \
    --train-augmentation-order "${AUG_ORDER}" \
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
    --data-dir "${DATA_DIR}" \
    --output-dir "${PWD}/${OUTPUT_ROOT}"

echo "Experiment ${EXP_NAME} completed successfully."