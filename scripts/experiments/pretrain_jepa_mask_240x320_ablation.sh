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
if [[ -z "${MPX_ROOT:-}" ]]; then
  if [[ "${TBIN}" == "1" ]]; then
    MPX_ROOT="/media/apollo-22/AT_2TB/dataset/t_1/1mpx_voxels_20s_tbin1"
  else
    MPX_ROOT="/mnt/data/arata/t_10/1mpx_voxels_20s"
  fi
fi
if [[ -z "${EVENTSCAPE_ROOT:-}" ]]; then
  if [[ "${TBIN}" == "1" ]]; then
    EVENTSCAPE_ROOT="/media/apollo-22/AT_2TB/dataset/t_1/EventScape_voxels_tbin1/Town01-03_train"
  else
    EVENTSCAPE_ROOT="/mnt/data/arata/t_10/EventScape_voxels/Town01-03_train"
  fi
fi

DATASETS="[${DSEC_ROOT}/train,${MPX_ROOT}/train,${EVENTSCAPE_ROOT}]"
DATASET_FPCS="${DATASET_FPCS:-[8,8,8]}"
DATASET_WEIGHTS="${DATASET_WEIGHTS:-[0.15,0.7,0.15]}"

RUN_RANDOM="${RUN_RANDOM:-1}"
RUN_ADAPTIVE_AREA="${RUN_ADAPTIVE_AREA:-1}"
RUN_STRATEGIC="${RUN_STRATEGIC:-1}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-2000}"
PRETRAIN_IPE_SCALE="${PRETRAIN_IPE_SCALE:-1.25}"
PRETRAIN_START_LR="${PRETRAIN_START_LR:-1.0e-6}"
PRETRAIN_LR="${PRETRAIN_LR:-1.0e-4}"
PRETRAIN_FINAL_LR="${PRETRAIN_FINAL_LR:-1.0e-6}"
PRETRAIN_WARMUP="${PRETRAIN_WARMUP:-40}"
PRETRAIN_NUM_WORKERS="${PRETRAIN_NUM_WORKERS:-16}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-64}"
VIS_INTERVAL="${VIS_INTERVAL:-0}"
VIS_MAX_TEMPORAL_SLICES="${VIS_MAX_TEMPORAL_SLICES:-8}"
PRETRAIN_SAVE_EVERY_FREQ="${PRETRAIN_SAVE_EVERY_FREQ:-250}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
RUN_PREFIX="${RUN_PREFIX:-stage1_jepa_mask}"
PREDICT_ALL="${PREDICT_ALL:-false}"
LAMBDA_VALUE_VID="${LAMBDA_VALUE_VID:-0.0}"
LAMBDA_PROGRESSIVE="${LAMBDA_PROGRESSIVE:-true}"
WEIGHT_DISTANCE_LOSS="${WEIGHT_DISTANCE_LOSS:-false}"
TENSORBOARD="${TENSORBOARD:-true}"

if [[ "${TBIN}" == "1" ]]; then
  RUN_SUFFIX="1gpu_mix_${IN_CHANS}ch_240x320_w015_070_015"
else
  RUN_SUFFIX="1gpu_mix_${IN_CHANS}ch_240x320_tbin${TBIN}_w015_070_015"
fi

COMMON_ARGS=(
  "data.datasets=${DATASETS}"
  "data.dataset_fpcs=${DATASET_FPCS}"
  "data.datasets_weights=${DATASET_WEIGHTS}"
  "data.batch_size=${PRETRAIN_BATCH_SIZE}"
  "data.num_workers=${PRETRAIN_NUM_WORKERS}"
  "data.crop_size=[240,320]"
  "data_aug.preserve_input_size=false"
  "data_aug.pad_to_hw=null"
  "data_aug.allowed_input_hw=null"
  "data_aug.random_resize_scale=[1.0,1.0]"
  "data_aug.random_resize_aspect_ratio=[1.3333333333,1.3333333333]"
  "model=vit_tiny_2_1"
  "model.in_chans=${IN_CHANS}"
  "model.lambda_value_vid=${LAMBDA_VALUE_VID}"
  "model.lambda_progressive=${LAMBDA_PROGRESSIVE}"
  "loss.predict_all=${PREDICT_ALL}"
  "loss.weight_distance_loss=${WEIGHT_DISTANCE_LOSS}"
  "optimization.epochs=${PRETRAIN_EPOCHS}"
  "optimization.ipe_scale=${PRETRAIN_IPE_SCALE}"
  "optimization.start_lr=${PRETRAIN_START_LR}"
  "optimization.lr=${PRETRAIN_LR}"
  "optimization.final_lr=${PRETRAIN_FINAL_LR}"
  "optimization.warmup=${PRETRAIN_WARMUP}"
  "optimization.clip_grad=1.0"
  "meta.use_tqdm=true"
  "meta.tensorboard=${TENSORBOARD}"
  "meta.vis_interval=${VIS_INTERVAL}"
  "meta.vis_max_temporal_slices=${VIS_MAX_TEMPORAL_SLICES}"
  "meta.save_every_freq=${PRETRAIN_SAVE_EVERY_FREQ}"
)

EXTRA_ARGS=("$@")

run_jepa() {
  local name="$1"
  local mask_name="$2"
  shift 2
  local folder_name="${RUN_PREFIX}_${name}_${RUN_SUFFIX}"
  echo "==== Running ${folder_name} mask=${mask_name} ===="
  python3 scripts/train/run_train.py \
    "folder=${OUTPUT_ROOT}/${folder_name}" \
    "${COMMON_ARGS[@]}" \
    "mask=${mask_name}" \
    "$@" \
    "${EXTRA_ARGS[@]}"
}

if [[ "${RUN_RANDOM}" == "1" ]]; then
  run_jepa "01_random" "stage1_event_random"
fi

if [[ "${RUN_ADAPTIVE_AREA}" == "1" ]]; then
  run_jepa "02_adaptive_area" "stage1_event_activity_adaptive"
fi

if [[ "${RUN_STRATEGIC}" == "1" ]]; then
  run_jepa "03_strategic" "stage1_event_strategic"
fi
