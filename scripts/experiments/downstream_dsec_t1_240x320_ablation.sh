#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DSEC_ROOT="${DSEC_ROOT:-/media/apollo-22/AT_2TB/dataset/t_1/DSEC_voxels_semantic_20s_tbin1}"

JEPA_FOLDER="${JEPA_FOLDER:-outputs/stage1_jepa_1gpu_mix_2ch_240x320_w015_070_015}"
MAE_FOLDER="${MAE_FOLDER:-outputs/stage1_mae_1gpu_mix_2ch_240x320_w015_070_015}"

JEPA_CKPT="${JEPA_CKPT:-}"
MAE_CKPT="${MAE_CKPT:-}"

DOWNSTREAM_EPOCHS="${DOWNSTREAM_EPOCHS:-100}"
DOWNSTREAM_CLIP_NUM_FRAMES="${DOWNSTREAM_CLIP_NUM_FRAMES:-8}"
DOWNSTREAM_NUM_WORKERS="${DOWNSTREAM_NUM_WORKERS:-16}"

RUN_JEPA_LINEAR="${RUN_JEPA_LINEAR:-1}"
RUN_JEPA_FINETUNE="${RUN_JEPA_FINETUNE:-1}"
RUN_SCRATCH="${RUN_SCRATCH:-1}"
RUN_MAE_LINEAR="${RUN_MAE_LINEAR:-1}"
RUN_MAE_FINETUNE="${RUN_MAE_FINETUNE:-1}"

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

echo "JEPA_CKPT=${JEPA_CKPT}"
echo "MAE_CKPT=${MAE_CKPT}"

COMMON_TASK_ARGS=(
  "task.train_roots=[${DSEC_ROOT}/train]"
  "task.val_roots=[${DSEC_ROOT}/test]"
  "task.input_size=[240,320]"
  "task.eval_original_resolution=true"
  "task.eval_logits_resize_mode=nearest"
  "task.clip_num_frames=${DOWNSTREAM_CLIP_NUM_FRAMES}"
  "task.num_workers=${DOWNSTREAM_NUM_WORKERS}"
  "model.num_frames=${DOWNSTREAM_CLIP_NUM_FRAMES}"
  "optimization.epochs=${DOWNSTREAM_EPOCHS}"
)

if [[ "${RUN_JEPA_LINEAR}" == "1" ]]; then
  python3 scripts/downstream/run_downstream.py \
    "folder=outputs/downstream_dsec_t1_vit_tiny_jepa_linear_probe_w015_070_015" \
    "model=vit_tiny_linear_probe" \
    "model.checkpoint_path=${JEPA_CKPT}" \
    "model.checkpoint_key=encoder" \
    "model.freeze_encoder=true" \
    "${COMMON_TASK_ARGS[@]}" \
    "task.batch_size=16"
fi

if [[ "${RUN_JEPA_FINETUNE}" == "1" ]]; then
  python3 scripts/downstream/run_downstream.py \
    "folder=outputs/downstream_dsec_t1_vit_tiny_jepa_finetune_w015_070_015" \
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
    "folder=outputs/downstream_dsec_t1_vit_tiny_scratch_w015_070_015" \
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
    "folder=outputs/downstream_dsec_t1_vit_tiny_mae_linear_probe_w015_070_015" \
    "model=vit_tiny_linear_probe" \
    "model.checkpoint_path=${MAE_CKPT}" \
    "model.checkpoint_key=encoder" \
    "model.freeze_encoder=true" \
    "${COMMON_TASK_ARGS[@]}" \
    "task.batch_size=16"
fi

if [[ "${RUN_MAE_FINETUNE}" == "1" ]]; then
  python3 scripts/downstream/run_downstream.py \
    "folder=outputs/downstream_dsec_t1_vit_tiny_mae_finetune_w015_070_015" \
    "model=vit_tiny_linear_probe" \
    "model.checkpoint_path=${MAE_CKPT}" \
    "model.checkpoint_key=encoder" \
    "model.freeze_encoder=false" \
    "${COMMON_TASK_ARGS[@]}" \
    "task.batch_size=8" \
    "optimization.lr=1.0e-3" \
    "optimization.encoder_lr=1.0e-5"
fi
