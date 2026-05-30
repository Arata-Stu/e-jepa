#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

M3ED_ROOT="${M3ED_ROOT:-/media/apollo-22/AT_2TB/dataset/t_1/m3ed_voxels_semantic_20s_tbin1}"
M3ED_TRAIN_ROOT="${M3ED_TRAIN_ROOT:-${M3ED_ROOT}/train}"
M3ED_VAL_ROOT="${M3ED_VAL_ROOT:-}"
if [[ -z "${M3ED_VAL_ROOT}" ]]; then
  if [[ -d "${M3ED_ROOT}/test" ]]; then
    M3ED_VAL_ROOT="${M3ED_ROOT}/test"
  else
    M3ED_VAL_ROOT="${M3ED_ROOT}/val"
  fi
fi

JEPA_FOLDER="${JEPA_FOLDER:-outputs/stage1_jepa_1gpu_mix_2ch_240x320_w015_070_015}"
MAE_FOLDER="${MAE_FOLDER:-outputs/stage1_mae_1gpu_mix_2ch_240x320_w015_070_015}"

JEPA_CKPT="${JEPA_CKPT:-}"
MAE_CKPT="${MAE_CKPT:-}"

DOWNSTREAM_EPOCHS="${DOWNSTREAM_EPOCHS:-100}"
DOWNSTREAM_CLIP_NUM_FRAMES="${DOWNSTREAM_CLIP_NUM_FRAMES:-8}"

RUN_JEPA_LINEAR="${RUN_JEPA_LINEAR:-1}"
RUN_JEPA_FINETUNE="${RUN_JEPA_FINETUNE:-1}"
RUN_SCRATCH="${RUN_SCRATCH:-1}"
RUN_MAE_LINEAR="${RUN_MAE_LINEAR:-1}"
RUN_MAE_FINETUNE="${RUN_MAE_FINETUNE:-1}"

if [[ ! -d "${M3ED_TRAIN_ROOT}" || ! -d "${M3ED_VAL_ROOT}" ]]; then
  echo "M3ED train/val roots were not found." >&2
  echo "  M3ED_TRAIN_ROOT=${M3ED_TRAIN_ROOT}" >&2
  echo "  M3ED_VAL_ROOT=${M3ED_VAL_ROOT}" >&2
  echo "Set it explicitly, for example:" >&2
  echo "  M3ED_ROOT=/path/to/m3ed_voxels_semantic_20s_tbin1 $0" >&2
  echo "or:" >&2
  echo "  M3ED_TRAIN_ROOT=/path/to/train M3ED_VAL_ROOT=/path/to/val $0" >&2
  exit 1
fi

NEED_JEPA=0
NEED_MAE=0
if [[ "${RUN_JEPA_LINEAR}" == "1" || "${RUN_JEPA_FINETUNE}" == "1" ]]; then
  NEED_JEPA=1
fi
if [[ "${RUN_MAE_LINEAR}" == "1" || "${RUN_MAE_FINETUNE}" == "1" ]]; then
  NEED_MAE=1
fi

if [[ "${NEED_JEPA}" == "1" && -z "${JEPA_CKPT}" && -d "${JEPA_FOLDER}" ]]; then
  JEPA_CKPT="$(find "${JEPA_FOLDER}" -name latest.pth.tar -print | sort | tail -n 1)"
fi

if [[ "${NEED_MAE}" == "1" && -z "${MAE_CKPT}" && -d "${MAE_FOLDER}" ]]; then
  MAE_CKPT="$(find "${MAE_FOLDER}" -name latest_mae.pth.tar -print | sort | tail -n 1)"
fi

if [[ "${NEED_JEPA}" == "1" && -z "${JEPA_CKPT}" ]]; then
  echo "JEPA checkpoint was not found under ${JEPA_FOLDER}" >&2
  exit 1
fi

if [[ "${NEED_MAE}" == "1" && -z "${MAE_CKPT}" ]]; then
  echo "MAE checkpoint was not found under ${MAE_FOLDER}" >&2
  exit 1
fi

echo "M3ED_ROOT=${M3ED_ROOT}"
echo "M3ED_TRAIN_ROOT=${M3ED_TRAIN_ROOT}"
echo "M3ED_VAL_ROOT=${M3ED_VAL_ROOT}"
echo "JEPA_CKPT=${JEPA_CKPT}"
echo "MAE_CKPT=${MAE_CKPT}"

COMMON_TASK_ARGS=(
  "task=m3ed_semantic"
  "task.train_roots=[${M3ED_TRAIN_ROOT}]"
  "task.val_roots=[${M3ED_VAL_ROOT}]"
  "task.input_size=[240,320]"
  "task.eval_original_resolution=true"
  "task.eval_logits_resize_mode=nearest"
  "task.clip_num_frames=${DOWNSTREAM_CLIP_NUM_FRAMES}"
  "model.num_frames=${DOWNSTREAM_CLIP_NUM_FRAMES}"
  "optimization.epochs=${DOWNSTREAM_EPOCHS}"
)

if [[ "${RUN_JEPA_LINEAR}" == "1" ]]; then
  python3 scripts/downstream/run_downstream.py \
    "folder=outputs/downstream_m3ed_sem_t1_vit_tiny_jepa_linear_probe_w015_070_015" \
    "model=vit_tiny_linear_probe" \
    "model.checkpoint_path=${JEPA_CKPT}" \
    "model.checkpoint_key=encoder" \
    "model.freeze_encoder=true" \
    "${COMMON_TASK_ARGS[@]}" \
    "task.batch_size=16"
fi

if [[ "${RUN_JEPA_FINETUNE}" == "1" ]]; then
  python3 scripts/downstream/run_downstream.py \
    "folder=outputs/downstream_m3ed_sem_t1_vit_tiny_jepa_finetune_w015_070_015" \
    "model=vit_tiny_linear_probe" \
    "model.checkpoint_path=${JEPA_CKPT}" \
    "model.checkpoint_key=encoder" \
    "model.freeze_encoder=false" \
    "${COMMON_TASK_ARGS[@]}" \
    "task.batch_size=8" \
    "optimization.lr=1.0e-3" \
    "optimization.encoder_lr=1.0e-5"
fi

if [[ "${RUN_SCRATCH}" == "1" ]]; then
  python3 scripts/downstream/run_downstream.py \
    "folder=outputs/downstream_m3ed_sem_t1_vit_tiny_scratch_w015_070_015" \
    "model=vit_tiny_linear_probe" \
    "model.checkpoint_path=null" \
    "model.freeze_encoder=false" \
    "${COMMON_TASK_ARGS[@]}" \
    "task.batch_size=8" \
    "optimization.lr=1.0e-3" \
    "optimization.encoder_lr=3.0e-4"
fi

if [[ "${RUN_MAE_LINEAR}" == "1" ]]; then
  python3 scripts/downstream/run_downstream.py \
    "folder=outputs/downstream_m3ed_sem_t1_vit_tiny_mae_linear_probe_w015_070_015" \
    "model=vit_tiny_linear_probe" \
    "model.checkpoint_path=${MAE_CKPT}" \
    "model.checkpoint_key=encoder" \
    "model.freeze_encoder=true" \
    "${COMMON_TASK_ARGS[@]}" \
    "task.batch_size=16"
fi

if [[ "${RUN_MAE_FINETUNE}" == "1" ]]; then
  python3 scripts/downstream/run_downstream.py \
    "folder=outputs/downstream_m3ed_sem_t1_vit_tiny_mae_finetune_w015_070_015" \
    "model=vit_tiny_linear_probe" \
    "model.checkpoint_path=${MAE_CKPT}" \
    "model.checkpoint_key=encoder" \
    "model.freeze_encoder=false" \
    "${COMMON_TASK_ARGS[@]}" \
    "task.batch_size=8" \
    "optimization.lr=1.0e-3" \
    "optimization.encoder_lr=1.0e-5"
fi
