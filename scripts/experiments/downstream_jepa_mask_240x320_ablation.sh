#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TBIN="${TBIN:-1}"
if [[ -z "${IN_CHANS:-}" ]]; then
  IN_CHANS=$((2 * TBIN))
fi

DOWNSTREAM_TASK="${DOWNSTREAM_TASK:-dsec_semantic}"

if [[ -z "${DSEC_ROOT:-}" ]]; then
  if [[ "${TBIN}" == "1" ]]; then
    DSEC_ROOT="/media/apollo-22/AT_2TB/dataset/t_1/DSEC_voxels_semantic_20s_tbin1"
  else
    DSEC_ROOT="/mnt/data/arata/t_10/DSEC_voxels_semantic_20s"
  fi
fi

if [[ -z "${M3ED_ROOT:-}" ]]; then
  if [[ "${TBIN}" == "1" ]]; then
    M3ED_ROOT="/media/apollo-22/AT_2TB/dataset/t_1/m3ed_voxels_semantic_20s_tbin1"
  else
    M3ED_ROOT="/mnt/data/arata/t_10/m3ed_voxels_semantic_20s"
  fi
fi
M3ED_TRAIN_ROOT="${M3ED_TRAIN_ROOT:-${M3ED_ROOT}/train}"
M3ED_VAL_ROOT="${M3ED_VAL_ROOT:-}"
if [[ -z "${M3ED_VAL_ROOT}" ]]; then
  if [[ -d "${M3ED_ROOT}/test" ]]; then
    M3ED_VAL_ROOT="${M3ED_ROOT}/test"
  else
    M3ED_VAL_ROOT="${M3ED_ROOT}/val"
  fi
fi

RUN_RANDOM="${RUN_RANDOM:-1}"
RUN_ADAPTIVE_AREA="${RUN_ADAPTIVE_AREA:-1}"
RUN_STRATEGIC="${RUN_STRATEGIC:-1}"
RUN_STRATEGIC_SMALL="${RUN_STRATEGIC_SMALL:-0}"
RUN_LINEAR="${RUN_LINEAR:-1}"
RUN_FINETUNE="${RUN_FINETUNE:-1}"

PRETRAIN_OUTPUT_ROOT="${PRETRAIN_OUTPUT_ROOT:-outputs}"
DOWNSTREAM_OUTPUT_ROOT="${DOWNSTREAM_OUTPUT_ROOT:-outputs}"
PRETRAIN_RUN_PREFIX="${PRETRAIN_RUN_PREFIX:-stage1_jepa_mask}"

RANDOM_CKPT="${RANDOM_CKPT:-}"
ADAPTIVE_AREA_CKPT="${ADAPTIVE_AREA_CKPT:-}"
STRATEGIC_CKPT="${STRATEGIC_CKPT:-}"
STRATEGIC_SMALL_CKPT="${STRATEGIC_SMALL_CKPT:-}"

DOWNSTREAM_EPOCHS="${DOWNSTREAM_EPOCHS:-100}"
DOWNSTREAM_CLIP_NUM_FRAMES="${DOWNSTREAM_CLIP_NUM_FRAMES:-8}"
DOWNSTREAM_NUM_WORKERS="${DOWNSTREAM_NUM_WORKERS:-16}"
DOWNSTREAM_LINEAR_BATCH_SIZE="${DOWNSTREAM_LINEAR_BATCH_SIZE:-16}"
DOWNSTREAM_FINETUNE_BATCH_SIZE="${DOWNSTREAM_FINETUNE_BATCH_SIZE:-8}"
DOWNSTREAM_TENSORBOARD="${DOWNSTREAM_TENSORBOARD:-true}"
DOWNSTREAM_SAVE_EVERY_FREQ="${DOWNSTREAM_SAVE_EVERY_FREQ:--1}"
DOWNSTREAM_CHECKPOINT_FREQ="${DOWNSTREAM_CHECKPOINT_FREQ:-1}"

if [[ "${TBIN}" == "1" ]]; then
  STAGE1_RUN_SUFFIX="1gpu_mix_${IN_CHANS}ch_240x320_w015_070_015"
  DSEC_RUN_TAG="dsec_t1"
  M3ED_RUN_TAG="m3ed_sem_t1"
else
  STAGE1_RUN_SUFFIX="1gpu_mix_${IN_CHANS}ch_240x320_tbin${TBIN}_w015_070_015"
  DSEC_RUN_TAG="dsec_tbin${TBIN}_${IN_CHANS}ch"
  M3ED_RUN_TAG="m3ed_sem_tbin${TBIN}_${IN_CHANS}ch"
fi
DOWNSTREAM_RUN_SUFFIX="w015_070_015"

case "${DOWNSTREAM_TASK}" in
  dsec_semantic)
    TASK_RUN_TAG="${DSEC_RUN_TAG}"
    TASK_ARGS=(
      "task=dsec_semantic"
      "task.train_roots=[${DSEC_ROOT}/train]"
      "task.val_roots=[${DSEC_ROOT}/test]"
    )
    ;;
  m3ed_semantic)
    TASK_RUN_TAG="${M3ED_RUN_TAG}"
    TASK_ARGS=(
      "task=m3ed_semantic"
      "task.train_roots=[${M3ED_TRAIN_ROOT}]"
      "task.val_roots=[${M3ED_VAL_ROOT}]"
    )
    ;;
  *)
    echo "Unsupported DOWNSTREAM_TASK=${DOWNSTREAM_TASK}. Use dsec_semantic or m3ed_semantic." >&2
    exit 1
    ;;
esac

COMMON_TASK_ARGS=(
  "${TASK_ARGS[@]}"
  "task.input_size=[240,320]"
  "task.eval_original_resolution=true"
  "task.eval_logits_resize_mode=nearest"
  "task.clip_num_frames=${DOWNSTREAM_CLIP_NUM_FRAMES}"
  "task.num_workers=${DOWNSTREAM_NUM_WORKERS}"
  "model.num_frames=${DOWNSTREAM_CLIP_NUM_FRAMES}"
  "model.in_chans=${IN_CHANS}"
  "optimization.epochs=${DOWNSTREAM_EPOCHS}"
  "meta.tensorboard=${DOWNSTREAM_TENSORBOARD}"
  "meta.save_every_freq=${DOWNSTREAM_SAVE_EVERY_FREQ}"
  "meta.checkpoint_freq=${DOWNSTREAM_CHECKPOINT_FREQ}"
)

EXTRA_ARGS=("$@")
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "hydra.run.dir=" ]]; then
    echo "hydra.run.dir is empty. Set it to a concrete output directory or omit it." >&2
    exit 1
  fi
done

find_stage1_checkpoint() {
  local strategy_name="$1"
  local explicit_ckpt="$2"
  local folder="${PRETRAIN_OUTPUT_ROOT}/${PRETRAIN_RUN_PREFIX}_${strategy_name}_${STAGE1_RUN_SUFFIX}"
  local ckpt=""

  if [[ -n "${explicit_ckpt}" ]]; then
    if [[ ! -f "${explicit_ckpt}" ]]; then
      echo "Explicit checkpoint does not exist: ${explicit_ckpt}" >&2
      exit 1
    fi
    echo "${explicit_ckpt}"
    return
  fi

  if [[ -d "${folder}" ]]; then
    ckpt="$(find "${folder}" -name latest.pth.tar -print | sort -V | tail -n 1)"
    if [[ -z "${ckpt}" ]]; then
      ckpt="$(find "${folder}" -name 'e*.pth.tar' -print | sort -V | tail -n 1)"
    fi
  fi

  if [[ -z "${ckpt}" ]]; then
    echo "Checkpoint was not found for ${strategy_name} under ${folder}" >&2
    echo "Set an explicit checkpoint, e.g. RANDOM_CKPT=/path/to/latest.pth.tar $0" >&2
    exit 1
  fi

  echo "${ckpt}"
}

run_downstream() {
  local strategy_label="$1"
  local ckpt="$2"
  local mode="$3"
  local freeze_encoder="$4"
  local batch_size="$5"
  shift 5

  local folder_name="downstream_${TASK_RUN_TAG}_vit_tiny_jepa_mask_${strategy_label}_${mode}_${DOWNSTREAM_RUN_SUFFIX}"
  echo "==== Running ${folder_name} ===="
  echo "checkpoint=${ckpt}"

  python3 scripts/downstream/run_downstream.py \
    "folder=${DOWNSTREAM_OUTPUT_ROOT}/${folder_name}" \
    "model=vit_tiny_linear_probe" \
    "model.checkpoint_path=${ckpt}" \
    "model.checkpoint_key=encoder" \
    "model.freeze_encoder=${freeze_encoder}" \
    "${COMMON_TASK_ARGS[@]}" \
    "task.batch_size=${batch_size}" \
    "$@" \
    "${EXTRA_ARGS[@]}"
}

run_strategy() {
  local strategy_name="$1"
  local strategy_label="$2"
  local explicit_ckpt="$3"
  local ckpt
  ckpt="$(find_stage1_checkpoint "${strategy_name}" "${explicit_ckpt}")"

  if [[ "${RUN_LINEAR}" == "1" ]]; then
    run_downstream \
      "${strategy_label}" \
      "${ckpt}" \
      "linear_probe" \
      "true" \
      "${DOWNSTREAM_LINEAR_BATCH_SIZE}"
  fi

  if [[ "${RUN_FINETUNE}" == "1" ]]; then
    run_downstream \
      "${strategy_label}" \
      "${ckpt}" \
      "finetune" \
      "false" \
      "${DOWNSTREAM_FINETUNE_BATCH_SIZE}" \
      "optimization.lr=1.0e-3" \
      "optimization.encoder_lr=1.0e-5"
  fi
}

echo "DOWNSTREAM_TASK=${DOWNSTREAM_TASK}"
echo "STAGE1_RUN_SUFFIX=${STAGE1_RUN_SUFFIX}"
echo "DOWNSTREAM_RUN_SUFFIX=${DOWNSTREAM_RUN_SUFFIX}"

if [[ "${RUN_RANDOM}" == "1" ]]; then
  run_strategy "01_random" "random" "${RANDOM_CKPT}"
fi

if [[ "${RUN_ADAPTIVE_AREA}" == "1" ]]; then
  run_strategy "02_adaptive_area" "adaptive_area" "${ADAPTIVE_AREA_CKPT}"
fi

if [[ "${RUN_STRATEGIC}" == "1" ]]; then
  run_strategy "03_strategic" "strategic" "${STRATEGIC_CKPT}"
fi

if [[ "${RUN_STRATEGIC_SMALL}" == "1" ]]; then
  run_strategy "04_strategic_small" "strategic_small" "${STRATEGIC_SMALL_CKPT}"
fi
