#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TBIN="${TBIN:-1}"
if [[ -z "${IN_CHANS:-}" ]]; then
  IN_CHANS=$((2 * TBIN))
fi

if [[ -z "${DSEC_ROOT:-}" ]]; then
  if [[ "${TBIN}" == "1" ]]; then
    DSEC_ROOT="/media/apollo-22/AT_2TB/dataset/t_1/DSEC_voxels_semantic_20s_tbin1"
  else
    DSEC_ROOT="/mnt/data/arata/t_10/DSEC_voxels_semantic_20s"
  fi
fi

if [[ -z "${JEPA_FOLDER:-}" ]]; then
  if [[ "${TBIN}" == "1" ]]; then
    JEPA_FOLDER="outputs/stage1_jepa_1gpu_mix_2ch_240x320_w015_070_015"
  else
    JEPA_FOLDER="outputs/stage1_jepa_1gpu_mix_${IN_CHANS}ch_240x320_tbin${TBIN}_w015_070_015"
  fi
fi
if [[ -z "${MAE_FOLDER:-}" ]]; then
  if [[ "${TBIN}" == "1" ]]; then
    MAE_FOLDER="outputs/stage1_mae_1gpu_mix_2ch_240x320_w015_070_015"
  else
    MAE_FOLDER="outputs/stage1_mae_1gpu_mix_${IN_CHANS}ch_240x320_tbin${TBIN}_w015_070_015"
  fi
fi

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
if [[ "${TBIN}" == "1" ]]; then
  DSEC_RUN_TAG="dsec_t1"
else
  DSEC_RUN_TAG="dsec_tbin${TBIN}_${IN_CHANS}ch"
fi
RUN_SUFFIX="w015_070_015"

COMMON_TASK_ARGS=(
  "task.train_roots=[${DSEC_ROOT}/train]"
  "task.val_roots=[${DSEC_ROOT}/test]"
  "task.input_size=[240,320]"
  "task.eval_original_resolution=true"
  "task.eval_logits_resize_mode=nearest"
  "task.clip_num_frames=${DOWNSTREAM_CLIP_NUM_FRAMES}"
  "task.num_workers=${DOWNSTREAM_NUM_WORKERS}"
  "model.num_frames=${DOWNSTREAM_CLIP_NUM_FRAMES}"
  "model.in_chans=${IN_CHANS}"
  "optimization.epochs=${DOWNSTREAM_EPOCHS}"
)

if [[ "${RUN_JEPA_LINEAR}" == "1" ]]; then
  python3 scripts/downstream/run_downstream.py \
    "folder=outputs/downstream_${DSEC_RUN_TAG}_vit_tiny_jepa_linear_probe_${RUN_SUFFIX}" \
    "model=vit_tiny_linear_probe" \
    "model.checkpoint_path=${JEPA_CKPT}" \
    "model.checkpoint_key=encoder" \
    "model.freeze_encoder=true" \
    "${COMMON_TASK_ARGS[@]}" \
    "task.batch_size=16"
fi

if [[ "${RUN_JEPA_FINETUNE}" == "1" ]]; then
  python3 scripts/downstream/run_downstream.py \
    "folder=outputs/downstream_${DSEC_RUN_TAG}_vit_tiny_jepa_finetune_${RUN_SUFFIX}" \
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
    "folder=outputs/downstream_${DSEC_RUN_TAG}_vit_tiny_scratch_${RUN_SUFFIX}" \
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
    "folder=outputs/downstream_${DSEC_RUN_TAG}_vit_tiny_mae_linear_probe_${RUN_SUFFIX}" \
    "model=vit_tiny_linear_probe" \
    "model.checkpoint_path=${MAE_CKPT}" \
    "model.checkpoint_key=encoder" \
    "model.freeze_encoder=true" \
    "${COMMON_TASK_ARGS[@]}" \
    "task.batch_size=16"
fi

if [[ "${RUN_MAE_FINETUNE}" == "1" ]]; then
  python3 scripts/downstream/run_downstream.py \
    "folder=outputs/downstream_${DSEC_RUN_TAG}_vit_tiny_mae_finetune_${RUN_SUFFIX}" \
    "model=vit_tiny_linear_probe" \
    "model.checkpoint_path=${MAE_CKPT}" \
    "model.checkpoint_key=encoder" \
    "model.freeze_encoder=false" \
    "${COMMON_TASK_ARGS[@]}" \
    "task.batch_size=8" \
    "optimization.lr=1.0e-3" \
    "optimization.encoder_lr=1.0e-5"
fi
