#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

M3ED_ROOT="${M3ED_ROOT:-/mnt/ssd-4tb/dataset/m3ed}"
M3ED_TRAIN_ROOT="${M3ED_TRAIN_ROOT:-${M3ED_ROOT}}"
M3ED_VAL_ROOT="${M3ED_VAL_ROOT:-${M3ED_ROOT}}"

STOPGRAD_FOLDER="${STOPGRAD_FOLDER:-outputs/m3ed_raw_jepa_smallmask_stopgrad_ema}"
SIGREG_FOLDER="${SIGREG_FOLDER:-outputs/m3ed_raw_jepa_smallmask_sigreg_w0.09}"
STOPGRAD_CKPT="${STOPGRAD_CKPT:-}"
SIGREG_CKPT="${SIGREG_CKPT:-}"

RUN_STOPGRAD="${RUN_STOPGRAD:-1}"
RUN_SIGREG="${RUN_SIGREG:-1}"

DOWNSTREAM_EPOCHS="${DOWNSTREAM_EPOCHS:-100}"
DOWNSTREAM_BATCH_SIZE="${DOWNSTREAM_BATCH_SIZE:-4}"
DOWNSTREAM_NUM_WORKERS="${DOWNSTREAM_NUM_WORKERS:-8}"
DOWNSTREAM_FREEZE_ENCODER="${DOWNSTREAM_FREEZE_ENCODER:-true}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
RUN_PREFIX="${RUN_PREFIX:-downstream_m3ed_raw_semantic}"

if [[ "${RUN_STOPGRAD}" == "1" && -z "${STOPGRAD_CKPT}" && -d "${STOPGRAD_FOLDER}" ]]; then
  STOPGRAD_CKPT="$(find "${STOPGRAD_FOLDER}" -name latest.pth.tar -print | sort | tail -n 1)"
fi

if [[ "${RUN_SIGREG}" == "1" && -z "${SIGREG_CKPT}" && -d "${SIGREG_FOLDER}" ]]; then
  SIGREG_CKPT="$(find "${SIGREG_FOLDER}" -name latest.pth.tar -print | sort | tail -n 1)"
fi

if [[ "${RUN_STOPGRAD}" == "1" && -z "${STOPGRAD_CKPT}" ]]; then
  echo "STOPGRAD_CKPT was not found. Set STOPGRAD_CKPT or STOPGRAD_FOLDER." >&2
  exit 1
fi

if [[ "${RUN_SIGREG}" == "1" && -z "${SIGREG_CKPT}" ]]; then
  echo "SIGREG_CKPT was not found. Set SIGREG_CKPT or SIGREG_FOLDER." >&2
  exit 1
fi

if [[ "${M3ED_TRAIN_ROOT}" == "${M3ED_VAL_ROOT}" ]]; then
  echo "M3ED_TRAIN_ROOT and M3ED_VAL_ROOT are identical." >&2
  echo "The task=m3ed_raw_semantic preset applies the F3 standard train/test sequence split by default." >&2
  echo "If you override task.*_sequence_ranges, keep train/val disjoint to avoid leakage." >&2
fi

COMMON_ARGS=(
  "task=m3ed_raw_semantic"
  "model=vit_tiny_m3ed_raw_linear_probe"
  "task.train_roots=[${M3ED_TRAIN_ROOT}]"
  "task.val_roots=[${M3ED_VAL_ROOT}]"
  "task.batch_size=${DOWNSTREAM_BATCH_SIZE}"
  "task.num_workers=${DOWNSTREAM_NUM_WORKERS}"
  "model.freeze_encoder=${DOWNSTREAM_FREEZE_ENCODER}"
  "optimization.epochs=${DOWNSTREAM_EPOCHS}"
)

EXTRA_ARGS=("$@")

run_downstream() {
  local run_name="$1"
  local checkpoint_path="$2"
  python3 scripts/downstream/run_downstream.py \
    "folder=${OUTPUT_ROOT}/${RUN_PREFIX}_${run_name}" \
    "model.checkpoint_path=${checkpoint_path}" \
    "model.checkpoint_key=encoder" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
}

echo "M3ED_TRAIN_ROOT=${M3ED_TRAIN_ROOT}"
echo "M3ED_VAL_ROOT=${M3ED_VAL_ROOT}"
echo "STOPGRAD_CKPT=${STOPGRAD_CKPT}"
echo "SIGREG_CKPT=${SIGREG_CKPT}"

if [[ "${RUN_STOPGRAD}" == "1" ]]; then
  run_downstream "stopgrad_ema" "${STOPGRAD_CKPT}"
fi

if [[ "${RUN_SIGREG}" == "1" ]]; then
  run_downstream "sigreg" "${SIGREG_CKPT}"
fi
